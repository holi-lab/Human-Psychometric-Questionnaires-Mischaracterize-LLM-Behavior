#!/usr/bin/env python3
"""
Established-questionnaire administration on cue-reduced surveys.

Copied and adapted from RQ1/established.py. Differences:
  - Surveys point at RQ3/cue_reduction/surveys_cueless/{PVQ,BFI44}.json
  - Prompt templates are the ORIGINAL prompts/*.txt (frames unchanged)
  - Output to RQ3/cue_reduction/results/established_cueless/{survey}/{model}.json
  - vLLM port 8011
  - 8 models: six TP1 models plus two large TP4 models
  - CLI: --limit N (first N items, for debug), --models a,b, --surveys PVQ,BFI44

Run with enough free GPUs for the selected models:
    python RQ3/cue_reduction/scripts/run_established_cueless.py [--limit 5 --models gemma-3-4b-it]
"""

import argparse
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

CUE_DIR = Path(__file__).resolve().parent.parent
ROOT = CUE_DIR.parent.parent  # Official/ repo root
CUELESS_DIR = CUE_DIR / "surveys_cueless"
OUT_ROOT = CUE_DIR / "results" / "established_cueless"


def log(msg: str = ""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Models ───────────────────────────────────────────────────────────────────
# Keep the six smaller models at TP1 to avoid the vLLM TP2 engine-init issue.
# The two large Qwen checkpoints use the known-good TP4 settings from RQ1.
ALL_MODELS = [
    {"name": "google/gemma-3-4b-it", "tensor_parallel": 1, "max_model_len": 2048},
    {"name": "google/gemma-3-27b-it", "tensor_parallel": 1, "max_model_len": 2048,
     "gpu_mem_util": 0.90},
    {"name": "Qwen/Qwen2.5-7B-Instruct", "tensor_parallel": 1, "max_model_len": 2048},
    {"name": "Qwen/Qwen3-30B-A3B-Instruct-2507", "tensor_parallel": 1,
     "max_model_len": 2048, "enforce_eager": True, "gpu_mem_util": 0.92},
    {"name": "openai/gpt-oss-20b", "tensor_parallel": 1, "max_model_len": 2048,
     "gpu_mem_util": 0.90},
    {"name": "openai/gpt-oss-120b", "tensor_parallel": 1, "max_model_len": 2048,
     "enforce_eager": True, "gpu_mem_util": 0.96},
    {"name": "Qwen/Qwen2.5-72B-Instruct", "tensor_parallel": 4,
     "max_model_len": 2048, "gpu_mem_util": 0.85, "startup_timeout": 7200},
    {"name": "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8", "tensor_parallel": 4,
     "max_model_len": 2048, "enforce_eager": True, "gpu_mem_util": 0.85,
     "startup_timeout": 7200},
]

VLLM_PORT = 8011
GPU_MEMORY_UTILIZATION = 0.90
DTYPE = "bfloat16"

SERVER_LAUNCH_RETRIES = 3
CONSECUTIVE_FAIL_LIMIT = 5


class ServerCrashError(Exception):
    pass


SURVEYS = {
    "BFI44": {
        "survey_file": str(CUELESS_DIR / "BFI44.json"),
        "prompts": {"normal": "prompts/BFI44.txt", "reversed": "prompts/BFI44_reversed.txt"},
        "type": "bfi", "construct_key": "trait", "scale_max": 5,
    },
    "PVQ": {
        "survey_file": str(CUELESS_DIR / "PVQ.json"),
        "prompts": {"normal": "prompts/PVQ.txt", "reversed": "prompts/PVQ_reversed.txt"},
        "type": "pvq", "construct_key": "value", "scale_max": 6,
    },
}

BFI_SCALE = {
    "disagree strongly": 1, "disagree a little": 2,
    "neither agree nor disagree": 3, "agree a little": 4, "agree strongly": 5,
}
PVQ_SCALE = {
    "not like me at all": 1, "not like me": 2, "a little like me": 3,
    "somewhat like me": 4, "like me": 5, "very much like me": 6,
}
LIKERT_PARSE_RETRIES = 3
MAX_TOKENS = 1024
_THINK_RE = None

# runtime-configurable
LIMIT = None


# ── GPU helpers ──────────────────────────────────────────────────────────────
def get_free_gpus(threshold_mb: int = 1000) -> list[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"], text=True)
        free = []
        for line in out.strip().splitlines():
            idx, mem_used = line.split(",")
            if int(mem_used.strip()) < threshold_mb:
                free.append(int(idx.strip()))
        return free
    except Exception:
        return []


def model_slug(name: str) -> str:
    return name.split("/")[-1]


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


def launch_vllm(model_cfg: dict) -> subprocess.Popen:
    ensure_port_free(VLLM_PORT)
    tp = model_cfg.get("tensor_parallel", 1)

    # Under a cluster scheduler, CUDA_VISIBLE_DEVICES is already set to the allocated
    # GPUs — trust it rather than re-probing physical indices (which can mismatch).
    slurm_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    slurm_devices = [d for d in slurm_cvd.split(",") if d.strip() != ""]
    if slurm_devices:
        if len(slurm_devices) < tp:
            raise RuntimeError(
                f"Need {tp} GPU(s) for {model_cfg['name']}, "
                f"but the job allocation has {slurm_devices}")
        gpu_str = ",".join(slurm_devices[:tp])
    else:
        free_gpus = get_free_gpus()
        if len(free_gpus) < tp:
            raise RuntimeError(
                f"Need {tp} free GPU(s) for {model_cfg['name']}, "
                f"but only {len(free_gpus)} available: {free_gpus}")
        gpu_str = ",".join(str(g) for g in free_gpus[:tp])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_str
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["VLLM_LOGGING_LEVEL"] = "WARNING"

    gmu = model_cfg.get("gpu_mem_util", GPU_MEMORY_UTILIZATION)
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_cfg["name"],
        "--host", "0.0.0.0", "--port", str(VLLM_PORT),
        "--tensor-parallel-size", str(tp),
        "--max-model-len", str(model_cfg.get("max_model_len", 2048)),
        "--gpu-memory-utilization", str(gmu),
        "--dtype", DTYPE, "--trust-remote-code",
    ]
    if model_cfg.get("enforce_eager", False):
        cmd.append("--enforce-eager")

    log("=" * 60)
    log(f"  Launching: {model_cfg['name']}  GPUs: {gpu_str} (TP={tp})  Port: {VLLM_PORT}")
    log("=" * 60)

    log_dir = CUE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    vllm_log = log_dir / f"vllm_{model_slug(model_cfg['name'])}.log"
    log_fh = open(vllm_log, "w")
    return subprocess.Popen(cmd, env=env, stdout=log_fh,
                            stderr=subprocess.STDOUT, preexec_fn=os.setsid)


def wait_for_server(proc, port, timeout=3600, interval=10):
    url = f"http://localhost:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited code {proc.returncode} during startup")
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                log(f"  Server ready on port {port}")
                return
        except requests.RequestException:
            pass
        time.sleep(interval)
    raise TimeoutError(f"vLLM server on port {port} not ready after {timeout}s")


def kill_server(proc):
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


def launch_with_retry(model_cfg, max_retries=SERVER_LAUNCH_RETRIES):
    for attempt in range(max_retries):
        proc = None
        try:
            proc = launch_vllm(model_cfg)
            wait_for_server(
                proc, VLLM_PORT,
                timeout=model_cfg.get("startup_timeout", 3600),
            )
            return proc
        except (TimeoutError, Exception) as e:
            log(f"  Launch attempt {attempt + 1}/{max_retries} failed: {e}")
            if proc:
                kill_server(proc)
            if attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                log(f"  Waiting {wait}s before retry...")
                time.sleep(wait)
    raise RuntimeError(f"Failed to launch {model_cfg['name']} after {max_retries} attempts")


def _extract_reasoning(content, message):
    global _THINK_RE
    if _THINK_RE is None:
        import re
        _THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    reasoning = message.get("reasoning")
    m = _THINK_RE.search(content)
    if m:
        if not reasoning:
            reasoning = m.group(1).strip()
        content = _THINK_RE.sub("", content).strip()
    return content, reasoning or None


def query_model(model_name, messages, port=VLLM_PORT, max_retries=5, temperature=0.0):
    url = f"http://localhost:{port}/v1/chat/completions"
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    payload = {"model": model_name, "messages": messages,
               "max_tokens": MAX_TOKENS, "temperature": temperature}
    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            msg = data["choices"][0]["message"]
            raw = msg.get("content")
            if raw is None:
                raise ValueError("Model returned null content")
            content, reasoning = _extract_reasoning(raw, msg)
            return {"content": content, "reasoning": reasoning}
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                log(f"    Retry {attempt + 1}/{max_retries} (wait {wait}s): {e}")
                time.sleep(wait)
            else:
                raise


def parse_likert(response, scale):
    resp = response.strip().lower()
    if resp in scale:
        return scale[resp]
    for label in sorted(scale, key=len, reverse=True):
        if label in resp:
            return scale[label]
    return None


def query_and_parse_likert(model_name, prompt_text, scale, port=VLLM_PORT):
    reasoning = None
    raw_response = ""
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


def reverse_score(raw, scale_max):
    return scale_max + 1 - raw


def _aggregate(buckets):
    return {k: round(sum(v) / len(v), 4) if v else None
            for k, v in sorted(buckets.items())}


def compute_construct_averages(items):
    buckets = defaultdict(list)
    for item in items.values():
        if item["final_score"] is not None:
            buckets[item["construct"]].append(item["final_score"])
    return _aggregate(buckets)


def compute_higher_order_averages(items):
    buckets = defaultdict(list)
    for item in items.values():
        ho = item.get("higher_order_value")
        if ho and item["final_score"] is not None:
            buckets[ho].append(item["final_score"])
    return _aggregate(buckets)


def _combine_averages(a, b):
    keys = sorted(set(a) | set(b))
    out = {}
    for k in keys:
        vals = [d[k] for d in (a, b) if k in d and d[k] is not None]
        out[k] = round(sum(vals) / len(vals), 4) if vals else None
    return out


def run_survey(model_name, survey_name, survey_cfg, port=VLLM_PORT):
    survey_path = ROOT / survey_cfg["survey_file"]
    survey_data = json.loads(survey_path.read_text())

    scale = BFI_SCALE if survey_cfg["type"] == "bfi" else PVQ_SCALE
    scale_max = survey_cfg["scale_max"]
    construct_key = survey_cfg["construct_key"]

    item_ids = sorted(survey_data, key=lambda x: int(x) if str(x).isdigit() else str(x))
    if LIMIT is not None:
        item_ids = item_ids[:LIMIT]

    result = {
        "model": model_name, "survey": survey_name,
        "survey_type": survey_cfg["type"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_items": len(item_ids),
        "scale": dict(scale),
        "cue_removed": True,
        "responses": {}, "construct_averages": {},
    }
    if survey_cfg["type"] == "pvq":
        result["higher_order_averages"] = {}

    for prompt_key, prompt_file in survey_cfg["prompts"].items():
        prompt_path = ROOT / prompt_file
        if not prompt_path.exists():
            log(f"    WARNING: {prompt_file} not found, skipping.")
            continue
        template = prompt_path.read_text().strip()
        log(f"  [{survey_name}] prompt={prompt_key} ({len(item_ids)} items) ...")

        items_result = {}
        consecutive_fails = 0
        for item_id in item_ids:
            item_data = survey_data[item_id]
            content = item_data.get("en_neutral", "")
            meta = item_data.get("meta_data", {})
            construct = meta.get(construct_key, "unknown")
            scoring = meta.get("scoring", "normal")
            higher_order = meta.get("higher_order_value")

            prompt_text = template.replace("{content}", content)
            try:
                raw_response, raw_score, reasoning = query_and_parse_likert(
                    model_name, prompt_text, scale, port)
                consecutive_fails = 0
            except Exception as e:
                log(f"    #{item_id:>3} [ERR] query failed: {e}")
                raw_response, raw_score, reasoning = "", None, None
                consecutive_fails += 1
                if consecutive_fails >= CONSECUTIVE_FAIL_LIMIT:
                    raise ServerCrashError(
                        f"Server unresponsive ({consecutive_fails} consecutive failures)")

            if raw_score is not None and scoring == "reverse":
                final_score = reverse_score(raw_score, scale_max)
            else:
                final_score = raw_score

            entry = {
                "content": content, "construct": construct, "scoring": scoring,
                "raw_response": raw_response, "raw_score": raw_score,
                "final_score": final_score,
            }
            if reasoning:
                entry["reasoning"] = reasoning
            if higher_order:
                entry["higher_order_value"] = higher_order
            items_result[item_id] = entry

            mark = "OK" if raw_score is not None else "FAIL"
            log(f"    #{item_id:>3} [{mark}] raw={raw_score} final={final_score}  \"{raw_response[:50]}\"")

        result["responses"][prompt_key] = items_result
        result["construct_averages"][prompt_key] = compute_construct_averages(items_result)
        if survey_cfg["type"] == "pvq":
            result["higher_order_averages"][prompt_key] = compute_higher_order_averages(items_result)

    pkeys = [k for k in result["construct_averages"] if k != "combined"]
    if len(pkeys) == 2:
        result["construct_averages"]["combined"] = _combine_averages(
            result["construct_averages"][pkeys[0]], result["construct_averages"][pkeys[1]])
        if "higher_order_averages" in result:
            ho_a = result["higher_order_averages"].get(pkeys[0], {})
            ho_b = result["higher_order_averages"].get(pkeys[1], {})
            result["higher_order_averages"]["combined"] = _combine_averages(ho_a, ho_b)
    return result


def save_result(result, survey_name, model_name):
    out_dir = OUT_ROOT / survey_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_slug(model_name)}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log(f"  => Saved: {out_path}")


def main():
    global LIMIT
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only run first N items per survey (debug)")
    ap.add_argument("--models", type=str, default=None,
                    help="comma-separated model slugs or full names to include")
    ap.add_argument("--surveys", type=str, default="PVQ,BFI44",
                    help="comma-separated survey names")
    args = ap.parse_args()
    LIMIT = args.limit

    models = ALL_MODELS
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        models = [m for m in ALL_MODELS
                  if m["name"] in wanted or model_slug(m["name"]) in wanted]
    survey_names = [s.strip() for s in args.surveys.split(",")]

    log("=" * 60)
    log("  Established questionnaire on cue-reduced surveys")
    log(f"  Models: {[model_slug(m['name']) for m in models]}")
    log(f"  Surveys: {survey_names}  Limit: {LIMIT}")
    log("=" * 60)

    free_gpus = get_free_gpus()
    log(f"Free GPUs: {free_gpus}")
    if not free_gpus:
        log("ERROR: No free GPUs. Exiting.")
        sys.exit(1)

    for model_cfg in models:
        model_name = model_cfg["name"]
        tp = model_cfg.get("tensor_parallel", 1)
        free_gpus = get_free_gpus()
        if len(free_gpus) < tp:
            log(f"SKIP {model_name}: need {tp} GPU(s), only {len(free_gpus)} free.")
            continue

        survey_items = [(sn, SURVEYS[sn]) for sn in survey_names if sn in SURVEYS]

        for restart in range(SERVER_LAUNCH_RETRIES):
            proc = None
            try:
                proc = launch_with_retry(model_cfg)
                for survey_name, survey_cfg in survey_items:
                    log("─" * 50)
                    log(f"  Model: {model_name}  Survey: {survey_name}")
                    log("─" * 50)
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
