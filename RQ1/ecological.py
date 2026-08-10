#!/usr/bin/env python3
"""
Ecological Validity: VP Generation Probability Profile

Launches vLLM models one-by-one on available GPUs, computes the
length-normalized log-likelihood of each candidate response for every
VP scenario, and saves raw per-token log-probs for downstream analysis.

For each VP scenario (104 total, 5 candidate responses each):
  - Selects prompt template by portrait_id range:
      1000-2999 -> VP_advice.txt  (uses {title} and {text})
      3000-4999 -> VP_chat.txt    (uses {text} only)
  - Formats the prompt via the model's chat template
  - Computes log P(response | prompt) token-by-token using echo + logprobs
  - Stores: total_logprob, num_tokens, normalized_logprob, per-token arrays

Results are saved incrementally for crash recovery.

Usage:
    python RQ1/ecological.py

Results:
    results/RQ1/ecological/{model_slug}.json
"""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent


def log(msg: str = ""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ===========================================================
#  Model Configuration
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
    },
    {
        "name": "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
        "tensor_parallel": 4,
        "max_model_len": 2048,
    }
]

VLLM_PORT = 8000
GPU_MEMORY_UTILIZATION = 0.85
DTYPE = "bfloat16"

SERVER_LAUNCH_RETRIES = 3
CONSECUTIVE_FAIL_LIMIT = 5

VP_SURVEY = ROOT / "surveys" / "VP.json"
PROMPT_ADVICE = ROOT / "prompts" / "VP_advice.txt"
PROMPT_CHAT = ROOT / "prompts" / "VP_chat.txt"


class ServerCrashError(Exception):
    pass


# ---------------------------------------------------------------
#  GPU Helpers
# ---------------------------------------------------------------

def get_free_gpus(threshold_mb: int = 1000) -> list[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
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


# ---------------------------------------------------------------
#  vLLM Server Management
# ---------------------------------------------------------------

def launch_vllm(model_cfg: dict) -> subprocess.Popen:
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
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_cfg["name"],
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--tensor-parallel-size", str(tp),
        "--max-model-len", str(model_cfg.get("max_model_len", 2048)),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--dtype", DTYPE,
        "--trust-remote-code",
    ]

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

    return subprocess.Popen(
        cmd, env=env,
        stdout=log_fh, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


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


def wait_for_server(proc: subprocess.Popen, port: int,
                    timeout: int = 3600, interval: int = 10):
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


def launch_with_retry(model_cfg: dict,
                      max_retries: int = SERVER_LAUNCH_RETRIES) -> subprocess.Popen:
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


# ---------------------------------------------------------------
#  Prompt Formatting
# ---------------------------------------------------------------

def get_prompt_type(portrait_id: int) -> str:
    if 1000 <= portrait_id < 3000:
        return "advice"
    if 3000 <= portrait_id < 5000:
        return "chat"
    raise ValueError(f"Unexpected portrait_id: {portrait_id}")


def format_scenario_prompt(
    scenario_content: dict, portrait_id: int,
    advice_template: str, chat_template: str,
) -> str:
    ptype = get_prompt_type(portrait_id)
    if ptype == "advice":
        return (
            advice_template
            .replace("{title}", scenario_content["title"])
            .replace("{text}", scenario_content["text"])
        )
    return chat_template.replace("{text}", scenario_content["text"])


# ---------------------------------------------------------------
#  Log-Probability Computation
# ---------------------------------------------------------------

def _load_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def compute_response_logprob(
    model_name: str, user_prompt: str, response_text: str,
    tokenizer, port: int = VLLM_PORT, max_retries: int = 3,
) -> dict:
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=True, add_generation_prompt=True,
    )
    prompt_len = len(prompt_ids)

    formatted_full = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response_text},
        ],
        tokenize=False, add_generation_prompt=False,
    )

    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": model_name,
        "prompt": formatted_full,
        "max_tokens": 1,
        "temperature": 1.0,
        "logprobs": 1,
        "echo": True,
    }

    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=payload, timeout=300)
            r.raise_for_status()
            data = r.json()

            lp = data["choices"][0]["logprobs"]
            all_logprobs = lp["token_logprobs"]
            all_tokens = lp["tokens"]

            resp_logprobs = all_logprobs[prompt_len:-1]
            resp_tokens = all_tokens[prompt_len:-1]

            valid = [v for v in resp_logprobs if v is not None]
            total = sum(valid)
            n = len(valid)

            return {
                "total_logprob": round(total, 6),
                "num_tokens": n,
                "normalized_logprob": round(total / n, 6) if n > 0 else None,
                "token_logprobs": [
                    round(v, 6) if v is not None else None for v in resp_logprobs
                ],
                "tokens": resp_tokens,
            }
        except Exception as e:
            if attempt < max_retries - 1:
                log(f"    Retry {attempt + 1}/{max_retries}: {e}")
                time.sleep(5)
            else:
                log(f"    FAILED after {max_retries} attempts: {e}")
                return {
                    "total_logprob": None,
                    "num_tokens": 0,
                    "normalized_logprob": None,
                    "token_logprobs": [],
                    "tokens": [],
                    "error": str(e),
                }


# ---------------------------------------------------------------
#  I/O
# ---------------------------------------------------------------

def model_slug(name: str) -> str:
    return name.split("/")[-1]


def _result_path(model_name: str) -> Path:
    return ROOT / "results" / "RQ1" / "ecological" / f"{model_slug(model_name)}.json"


def save_result(result: dict, model_name: str):
    out_path = _result_path(model_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log(f"  => Saved: {out_path.relative_to(ROOT)}")


def _load_existing_result(model_name: str) -> dict | None:
    path = _result_path(model_name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------
#  VP Batch Runner
# ---------------------------------------------------------------

def run_vp_batch(
    model_name: str,
    scenarios: list[dict],
    batch_indices: list[int],
    tokenizer,
    result: dict,
    done_pids: set[int],
    port: int = VLLM_PORT,
):
    advice_tpl = PROMPT_ADVICE.read_text().strip()
    chat_tpl = PROMPT_CHAT.read_text().strip()
    total = len(scenarios)
    new_count = 0

    for idx in batch_indices:
        scenario = scenarios[idx]
        pid = scenario["portrait_id"]

        if pid in done_pids:
            continue

        ptype = get_prompt_type(pid)
        prompt_text = format_scenario_prompt(
            scenario["content"], pid, advice_tpl, chat_tpl,
        )

        log(f"  [{idx + 1}/{total}] portrait_id={pid} ({ptype})")

        outputs: list[dict] = []
        consecutive_fails = 0
        for out in scenario["outputs"]:
            lp_data = compute_response_logprob(
                model_name, prompt_text, out["content"], tokenizer, port,
            )
            outputs.append({
                "output_id": out["id"],
                "content": out["content"],
                **lp_data,
            })
            norm = lp_data.get("normalized_logprob")
            norm_s = f"{norm:.4f}" if norm is not None else "N/A"
            log(
                f"    output {out['id']}: "
                f"norm_logprob={norm_s}  tokens={lp_data['num_tokens']}"
            )

            if lp_data.get("error"):
                consecutive_fails += 1
                if consecutive_fails >= CONSECUTIVE_FAIL_LIMIT:
                    save_result(result, model_name)
                    raise ServerCrashError(
                        f"Server unresponsive ({consecutive_fails} consecutive failures)"
                    )
            else:
                consecutive_fails = 0

        result["results"].append({
            "portrait_id": pid,
            "prompt_type": ptype,
            "title": scenario["content"].get("title", ""),
            "outputs": outputs,
        })
        done_pids.add(pid)

        new_count += 1
        if new_count % 10 == 0:
            save_result(result, model_name)

    if new_count > 0:
        save_result(result, model_name)


# ---------------------------------------------------------------
#  Main
# ---------------------------------------------------------------

def main():
    log("=" * 60)
    log("  Ecological Validity: VP Log-Probability Profile")
    log("=" * 60)

    free_gpus = get_free_gpus()
    log(f"Free GPUs: {free_gpus}")
    if not free_gpus:
        log("ERROR: No free GPUs. Exiting.")
        sys.exit(1)

    scenarios = json.loads(VP_SURVEY.read_text())

    for model_cfg in MODELS:
        model_name = model_cfg["name"]
        tp = model_cfg.get("tensor_parallel", 1)

        free_gpus = get_free_gpus()
        if len(free_gpus) < tp:
            log(f"SKIP {model_name}: need {tp} GPU(s), only {len(free_gpus)} free.")
            continue

        log(f"{'─' * 50}")
        log(f"  Model: {model_name}")
        log(f"  Task: VP Log-Probability Profile")
        log(f"{'─' * 50}")

        log(f"  Loading tokenizer for {model_name} ...")
        tokenizer = _load_tokenizer(model_name)

        existing = _load_existing_result(model_name)
        done_pids: set[int] = set()
        if existing:
            done_pids = {r["portrait_id"] for r in existing.get("results", [])}
            log(f"  Resuming: {len(done_pids)}/{len(scenarios)} scenarios already done")

        remaining = [
            i for i in range(len(scenarios))
            if scenarios[i]["portrait_id"] not in done_pids
        ]
        if not remaining:
            log(f"  All {len(scenarios)} scenarios already done, skipping")
            continue

        result: dict = existing or {
            "model": model_name,
            "survey": "VP",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_scenarios": len(scenarios),
            "results": [],
        }

        proc = None
        for attempt in range(SERVER_LAUNCH_RETRIES):
            try:
                proc = launch_with_retry(model_cfg)
                run_vp_batch(
                    model_name, scenarios, list(range(len(scenarios))),
                    tokenizer, result, done_pids,
                )
                break

            except ServerCrashError as e:
                log(f"  SERVER CRASH: {e}")
                if proc:
                    kill_server(proc)
                    proc = None
                if attempt < SERVER_LAUNCH_RETRIES - 1:
                    log(f"  Restarting server (attempt {attempt + 2}) ...")
                    time.sleep(15)
                else:
                    log(f"  Giving up on {model_name} after {attempt + 1} restarts")

            except Exception as e:
                log(f"ERROR with {model_name}: {e}")
                import traceback
                traceback.print_exc()
                break
            finally:
                if proc:
                    kill_server(proc)
                    proc = None

        log(f"  {model_name} complete: {len(done_pids)}/{len(scenarios)} scenarios")

    log("=" * 60)
    log("  All done!")
    log("=" * 60)


if __name__ == "__main__":
    main()
