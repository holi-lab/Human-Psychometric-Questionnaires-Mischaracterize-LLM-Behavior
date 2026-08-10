#!/usr/bin/env python3
"""
Established Questionnaire Profile Builder

Launches vLLM models one-by-one on available GPUs, runs all configured
questionnaires (PVQ-40, PVQ-21, BFI-44, BFI-10) with normal & reversed
prompts, applies reverse scoring where tagged, computes construct-level
averages, and saves results.

Usage:
    python RQ1/established.py

Results:
    results/RQ1/established/{survey_name}/{model_slug}.json
"""

import json
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent


def log(msg: str = ""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ===========================================================
#  Model Configuration
#  - Models are served one at a time; all surveys run, then the server stops
#  - tensor_parallel: number of GPUs, matched to model size
#  - enforce_eager: for models (e.g. MoE) where CUDA graph compilation OOMs
# ===========================================================
MODELS = [
    {
        "name": "google/gemma-3-4b-it",
        "tensor_parallel": 1,
        "max_model_len": 2048,
    },
    {
        "name": "google/gemma-3-27b-it",
        "tensor_parallel": 2,
        "max_model_len": 2048,
    },
    {
        "name": "openai/gpt-oss-20b",
        "tensor_parallel": 2,
        "max_model_len": 2048,
    },
    {
        "name": "openai/gpt-oss-120b",
        "tensor_parallel": 4,
        "max_model_len": 2048,
        "enforce_eager": True,
    },
    {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "tensor_parallel": 1,
        "max_model_len": 2048,
    },
    {
        "name": "Qwen/Qwen2.5-72B-Instruct",
        "tensor_parallel": 4,
        "max_model_len": 2048,
    },
    {
        "name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "tensor_parallel": 2,
        "max_model_len": 2048,
        "enforce_eager": True,
    },
    {
        "name": "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
        "tensor_parallel": 4,
        "max_model_len": 2048,
        "enforce_eager": True,
    }
]

VLLM_PORT = 8001
GPU_MEMORY_UTILIZATION = 0.85
DTYPE = "bfloat16"

SERVER_LAUNCH_RETRIES = 3
CONSECUTIVE_FAIL_LIMIT = 5


class ServerCrashError(Exception):
    """Raised when too many consecutive failures indicate server crash."""
    pass


# ===========================================================
#  Survey Configuration
# ===========================================================
SURVEYS = {
    "BFI10": {
        "survey_file": "surveys/BFI10.json",
        "prompts": {
            "normal": "prompts/BFI10.txt",
            "reversed": "prompts/BFI10_reversed.txt",
        },
        "type": "bfi",
        "construct_key": "trait",
        "scale_max": 5,
    },
    "BFI44": {
        "survey_file": "surveys/BFI44.json",
        "prompts": {
            "normal": "prompts/BFI44.txt",
            "reversed": "prompts/BFI44_reversed.txt",
        },
        "type": "bfi",
        "construct_key": "trait",
        "scale_max": 5,
    },
    "PVQ": {
        "survey_file": "surveys/PVQ.json",
        "prompts": {
            "normal": "prompts/PVQ.txt",
            "reversed": "prompts/PVQ_reversed.txt",
        },
        "type": "pvq",
        "construct_key": "value",
        "scale_max": 6,
    },
    "PVQ21": {
        "survey_file": "surveys/PVQ21.json",
        "prompts": {
            "normal": "prompts/PVQ21.txt",
            "reversed": "prompts/PVQ21_reversed.txt",
        },
        "type": "pvq",
        "construct_key": "value",
        "scale_max": 6,
    },
}

# ===========================================================
#  Likert Scale Mappings (case-insensitive)
# ===========================================================
BFI_SCALE = {
    "disagree strongly": 1,
    "disagree a little": 2,
    "neither agree nor disagree": 3,
    "agree a little": 4,
    "agree strongly": 5,
}

PVQ_SCALE = {
    "not like me at all": 1,
    "not like me": 2,
    "a little like me": 3,
    "somewhat like me": 4,
    "like me": 5,
    "very much like me": 6,
}

LIKERT_PARSE_RETRIES = 3


# ───────────────────────────────────────────────────────────
#  GPU Helpers
# ───────────────────────────────────────────────────────────

def get_free_gpus(threshold_mb: int = 1000) -> list[int]:
    """Return indices of GPUs using less than *threshold_mb* MiB."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        free = []
        for line in out.strip().splitlines():
            idx, mem_used = line.split(",")
            if int(mem_used.strip()) < threshold_mb:
                free.append(int(idx.strip()))
        return free
    except Exception:
        return []


# ───────────────────────────────────────────────────────────
#  vLLM Server Management
# ───────────────────────────────────────────────────────────

def launch_vllm(model_cfg: dict) -> subprocess.Popen:
    """Start a vLLM OpenAI-compatible server and return the Popen handle."""
    ensure_port_free(VLLM_PORT)

    free_gpus = get_free_gpus()
    tp = model_cfg.get("tensor_parallel", 1)

    if len(free_gpus) < tp:
        raise RuntimeError(
            f"Need {tp} free GPU(s) for {model_cfg['name']}, "
            f"but only {len(free_gpus)} available: {free_gpus}"
        )

    selected_gpus = free_gpus[:tp]
    gpu_str = ",".join(str(g) for g in selected_gpus)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_str
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_cfg["name"],
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--tensor-parallel-size",
        str(tp),
        "--max-model-len",
        str(model_cfg.get("max_model_len", 2048)),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--dtype",
        DTYPE,
        "--trust-remote-code",
    ]

    if model_cfg.get("enforce_eager", False):
        cmd.append("--enforce-eager")

    log(f"{'=' * 60}")
    log(f"  Launching: {model_cfg['name']}")
    log(f"  GPUs: {gpu_str} (TP={tp})")
    log(f"  Port: {VLLM_PORT}")
    log(f"{'=' * 60}")

    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    vllm_log = log_dir / f"vllm_{model_slug(model_cfg['name'])}.log"
    log_fh = open(vllm_log, "w")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc


def ensure_port_free(port):
    """Abort if another process is already listening on the given port."""
    try:
        out = subprocess.check_output(
            ["ss", "-tln", f"sport = :{port}"],
            text=True,
        )
    except Exception:
        return
    if any(line.strip() for line in out.splitlines()[1:]):
        raise RuntimeError(
            f"Port {port} is already in use by another process. "
            "Free the port (or change VLLM_PORT) and re-run."
        )


def wait_for_server(proc: subprocess.Popen, port: int, timeout: int = 3600, interval: int = 10):
    """Block until the vLLM server responds to /v1/models.
    Raises early if the server process dies."""
    url = f"http://localhost:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"vLLM process exited with code {proc.returncode} during startup"
            )
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                log(f"  Server ready on port {port}")
                return
        except requests.RequestException:
            pass
        time.sleep(interval)
    raise TimeoutError(f"vLLM server on port {port} not ready after {timeout}s")


def kill_server(proc: subprocess.Popen):
    """Gracefully terminate the entire vLLM process group."""
    if proc.poll() is not None:
        return
    log("  Shutting down vLLM server...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
    log("  Server stopped.")


def launch_with_retry(model_cfg: dict, max_retries: int = SERVER_LAUNCH_RETRIES) -> subprocess.Popen:
    """Launch vLLM server with retries on startup failure."""
    for attempt in range(max_retries):
        proc = None
        try:
            proc = launch_vllm(model_cfg)
            wait_for_server(proc, VLLM_PORT)
            return proc
        except (TimeoutError, Exception) as e:
            log(f"  Launch attempt {attempt + 1}/{max_retries} failed: {e}")
            if proc:
                kill_server(proc)
            if attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                log(f"  Waiting {wait}s before retry...")
                time.sleep(wait)

    raise RuntimeError(
        f"Failed to launch {model_cfg['name']} after {max_retries} attempts"
    )


# ───────────────────────────────────────────────────────────
#  Model Query
# ───────────────────────────────────────────────────────────

MAX_TOKENS = 1024

_THINK_RE = None


def _extract_reasoning(content: str, message: dict) -> tuple[str, str | None]:
    """Extract reasoning and clean content.

    Returns (clean_content, reasoning_text_or_None).
    Sources:  gpt-oss  → message["reasoning"] field
              Qwen3    → <think>…</think> inside content
    """
    global _THINK_RE
    if _THINK_RE is None:
        import re
        _THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    reasoning = message.get("reasoning")

    think_match = _THINK_RE.search(content)
    if think_match:
        if not reasoning:
            reasoning = think_match.group(1).strip()
        content = _THINK_RE.sub("", content).strip()

    return content, reasoning or None


def query_model(
    model_name: str,
    messages: list[dict] | str,
    port: int = VLLM_PORT,
    max_retries: int = 5,
    temperature: float = 0.0,
) -> dict:
    """Send a chat prompt to vLLM and return parsed response.

    Returns {"content": str, "reasoning": str | None}.
    """
    url = f"http://localhost:{port}/v1/chat/completions"
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": temperature,
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            msg = data["choices"][0]["message"]
            raw_content = msg.get("content")
            if raw_content is None:
                raise ValueError("Model returned null content")
            content, reasoning = _extract_reasoning(raw_content, msg)
            return {"content": content, "reasoning": reasoning}
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                log(f"    Retry {attempt + 1}/{max_retries} (wait {wait}s): {e}")
                time.sleep(wait)
            else:
                raise


# ───────────────────────────────────────────────────────────
#  Scoring Helpers
# ───────────────────────────────────────────────────────────

def parse_likert(response: str, scale: dict[str, int]) -> int | None:
    """Parse a model response into a Likert integer. Returns None on failure."""
    resp = response.strip().lower()

    if resp in scale:
        return scale[resp]

    for label in sorted(scale, key=len, reverse=True):
        if label in resp:
            return scale[label]

    return None


def query_and_parse_likert(
    model_name: str,
    prompt_text: str,
    scale: dict[str, int],
    port: int = VLLM_PORT,
) -> tuple[str, int | None, str | None]:
    """Query the model and parse a Likert response, retrying the same
    prompt up to LIKERT_PARSE_RETRIES times on parse failure.

    Returns (raw_response, raw_score, reasoning).
    """
    reasoning = None
    for attempt in range(1 + LIKERT_PARSE_RETRIES):
        resp = query_model(model_name, prompt_text, port)
        raw_response = resp["content"]
        reasoning = resp["reasoning"]
        raw_score = parse_likert(raw_response, scale)
        if raw_score is not None:
            return raw_response, raw_score, reasoning
        if attempt < LIKERT_PARSE_RETRIES:
            log(f"          -> parse retry {attempt + 1}/{LIKERT_PARSE_RETRIES}: \"{raw_response[:60]}\"")

    return raw_response, None, reasoning


def reverse_score(raw: int, scale_max: int) -> int:
    return scale_max + 1 - raw


def _aggregate(buckets: dict[str, list[float]]) -> dict[str, float | None]:
    return {
        k: round(sum(v) / len(v), 4) if v else None
        for k, v in sorted(buckets.items())
    }


def compute_construct_averages(items: dict) -> dict[str, float | None]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for item in items.values():
        if item["final_score"] is not None:
            buckets[item["construct"]].append(item["final_score"])
    return _aggregate(buckets)


def compute_higher_order_averages(items: dict) -> dict[str, float | None]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for item in items.values():
        ho = item.get("higher_order_value")
        if ho and item["final_score"] is not None:
            buckets[ho].append(item["final_score"])
    return _aggregate(buckets)


def _combine_averages(avg_a: dict, avg_b: dict) -> dict[str, float | None]:
    """Average two dicts of construct → score, key-wise."""
    all_keys = sorted(set(avg_a) | set(avg_b))
    combined = {}
    for k in all_keys:
        vals = [d[k] for d in (avg_a, avg_b) if k in d and d[k] is not None]
        combined[k] = round(sum(vals) / len(vals), 4) if vals else None
    return combined


# ───────────────────────────────────────────────────────────
#  Survey Runner
# ───────────────────────────────────────────────────────────

def run_survey(
    model_name: str,
    survey_name: str,
    survey_cfg: dict,
    port: int = VLLM_PORT,
) -> dict:
    """Run one survey with both prompt variants. Returns the full result dict."""
    survey_path = ROOT / survey_cfg["survey_file"]
    survey_data = json.loads(survey_path.read_text())

    scale = BFI_SCALE if survey_cfg["type"] == "bfi" else PVQ_SCALE
    scale_max = survey_cfg["scale_max"]
    construct_key = survey_cfg["construct_key"]

    result = {
        "model": model_name,
        "survey": survey_name,
        "survey_type": survey_cfg["type"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_items": len(survey_data),
        "scale": dict(scale),
        "responses": {},
        "construct_averages": {},
    }

    if survey_cfg["type"] == "pvq":
        result["higher_order_averages"] = {}

    for prompt_key, prompt_file in survey_cfg["prompts"].items():
        prompt_path = ROOT / prompt_file
        if not prompt_path.exists():
            log(f"    WARNING: {prompt_file} not found, skipping.")
            continue

        template = prompt_path.read_text().strip()
        log(
            f"  [{survey_name}] prompt={prompt_key}  "
            f"({len(survey_data)} items) ..."
        )

        items_result: dict[str, dict] = {}
        consecutive_fails = 0
        for item_id, item_data in survey_data.items():
            content = item_data.get("en_neutral", "")
            meta = item_data.get("meta_data", {})
            construct = meta.get(construct_key, "unknown")
            scoring = meta.get("scoring", "normal")
            higher_order = meta.get("higher_order_value")

            prompt_text = template.replace("{content}", content)
            try:
                raw_response, raw_score, reasoning = query_and_parse_likert(
                    model_name, prompt_text, scale, port
                )
                consecutive_fails = 0
            except Exception as e:
                log(f"    #{item_id:>3} [ERR]  query failed: {e}")
                raw_response = ""
                raw_score = None
                reasoning = None
                consecutive_fails += 1
                if consecutive_fails >= CONSECUTIVE_FAIL_LIMIT:
                    raise ServerCrashError(
                        f"Server unresponsive ({consecutive_fails} consecutive failures)"
                    )

            if raw_score is not None and scoring == "reverse":
                final_score = reverse_score(raw_score, scale_max)
            else:
                final_score = raw_score

            entry: dict = {
                "content": content,
                "construct": construct,
                "scoring": scoring,
                "raw_response": raw_response,
                "raw_score": raw_score,
                "final_score": final_score,
            }
            if reasoning:
                entry["reasoning"] = reasoning
            if higher_order:
                entry["higher_order_value"] = higher_order

            items_result[item_id] = entry

            mark = "OK" if raw_score is not None else "FAIL"
            log(
                f"    #{item_id:>3} [{mark}] raw={raw_score} "
                f"final={final_score}  \"{raw_response[:60]}\""
            )

        result["responses"][prompt_key] = items_result
        result["construct_averages"][prompt_key] = compute_construct_averages(
            items_result
        )
        if survey_cfg["type"] == "pvq":
            result["higher_order_averages"][prompt_key] = (
                compute_higher_order_averages(items_result)
            )

    pkeys = [k for k in result["construct_averages"] if k != "combined"]
    if len(pkeys) == 2:
        result["construct_averages"]["combined"] = _combine_averages(
            result["construct_averages"][pkeys[0]],
            result["construct_averages"][pkeys[1]],
        )
        if "higher_order_averages" in result:
            ho_a = result["higher_order_averages"].get(pkeys[0], {})
            ho_b = result["higher_order_averages"].get(pkeys[1], {})
            result["higher_order_averages"]["combined"] = _combine_averages(
                ho_a, ho_b
            )

    return result


# ───────────────────────────────────────────────────────────
#  I/O
# ───────────────────────────────────────────────────────────

def model_slug(name: str) -> str:
    return name.split("/")[-1]


def save_result(result: dict, survey_name: str, model_name: str):
    out_dir = ROOT / "results" / "RQ1" / "established" / survey_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_slug(model_name)}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log(f"  => Saved: {out_path.relative_to(ROOT)}")


# ───────────────────────────────────────────────────────────
#  Main
# ───────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("  Established Questionnaire Profile Builder")
    log("=" * 60)

    free_gpus = get_free_gpus()
    log(f"Free GPUs: {free_gpus}")
    if not free_gpus:
        log("ERROR: No free GPUs. Exiting.")
        sys.exit(1)

    for model_cfg in MODELS:
        model_name = model_cfg["name"]
        tp = model_cfg.get("tensor_parallel", 1)

        free_gpus = get_free_gpus()
        if len(free_gpus) < tp:
            log(
                f"SKIP {model_name}: need {tp} GPU(s), "
                f"only {len(free_gpus)} free."
            )
            continue

        survey_items = [
            (sn, sc) for sn, sc in SURVEYS.items()
            if (ROOT / sc["survey_file"]).exists()
        ]
        if not survey_items:
            continue

        for restart in range(SERVER_LAUNCH_RETRIES):
            proc = None
            try:
                proc = launch_with_retry(model_cfg)

                for survey_name, survey_cfg in survey_items:
                    log(f"{'─' * 50}")
                    log(f"  Model: {model_name}")
                    log(f"  Survey: {survey_name}")
                    log(f"{'─' * 50}")

                    result = run_survey(model_name, survey_name, survey_cfg)
                    save_result(result, survey_name, model_name)

                break

            except ServerCrashError as e:
                log(f"  SERVER CRASH: {e}")
                if proc:
                    kill_server(proc)
                    proc = None
                if restart < SERVER_LAUNCH_RETRIES - 1:
                    log(f"  Restarting server (attempt {restart + 2}) ...")
                    time.sleep(15)
                else:
                    log(f"  Giving up on {model_name} after {restart + 1} restarts")

            except Exception as e:
                log(f"ERROR with {model_name}: {e}")
                import traceback
                traceback.print_exc()
                break
            finally:
                if proc:
                    kill_server(proc)

    log("=" * 60)
    log("  All done!")
    log("=" * 60)


if __name__ == "__main__":
    main()
