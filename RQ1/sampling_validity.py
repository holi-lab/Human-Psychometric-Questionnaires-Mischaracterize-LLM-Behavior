#!/usr/bin/env python3
"""
Sampling Log-Probability Validity

Validates that VP log-prob rankings are meaningful by comparing VP candidates'
log-probs against the distribution of log-probs from the model's own sampled
responses.

Rationale:
  If we sample K responses from the model at temperature=1 and compute their
  log-probs, these represent the "typical" range of log-probs the model
  assigns to its own natural outputs.  If the 5 VP candidate responses have
  log-probs that fall within (or near) this typical range, it indicates that
  the forced-choice log-prob evaluation is not operating in an anomalous
  region of the model's probability space — i.e., the setup is sound.

For each model and VP scenario:
  1. Sample K=10 free-form responses at temperature=1
  2. Compute log P(sampled_response | scenario_prompt) for each sample
  3. Load original VP candidate log-probs from RQ1 ecological results
  4. Compare VP log-probs against the sampled distribution

Metrics:
  - Percentile: where each VP candidate's normalized log-prob falls among
    the K sampled responses (0=below all, 100=above all)
  - VP rank-1 percentile: percentile of the VP top-1 candidate
  - Coverage rate: fraction of VP candidates within [min, max] of sampled
    log-probs
  - Z-score: (vp_logprob - mean_sampled) / std_sampled for each VP candidate
  - Mean sampled vs. mean VP log-probs

Usage:
    python RQ1/sampling_validity.py

Results:
    results/RQ1/sampling_validity/{model_slug}.json
    results/RQ1/sampling_validity/summary.json
"""

import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent

# ===========================================================
#  Model Configuration (identical to RQ1/ecological.py)
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
    },
]

VLLM_PORT = 8000
GPU_MEMORY_UTILIZATION = 0.85
DTYPE = "bfloat16"

SERVER_LAUNCH_RETRIES = 3
CONSECUTIVE_FAIL_LIMIT = 5
GENERATION_MAX_TOKENS = 512
NUM_SAMPLES = 10
SAMPLING_TEMPERATURE = 1.0

VP_SURVEY = ROOT / "surveys" / "VP.json"
PROMPT_ADVICE = ROOT / "prompts" / "VP_advice.txt"
PROMPT_CHAT = ROOT / "prompts" / "VP_chat.txt"

RQ1_DIR = ROOT / "results" / "RQ1" / "ecological"
OUT_DIR = ROOT / "results" / "RQ1" / "sampling_validity"


class ServerCrashError(Exception):
    pass


def log(msg: str = ""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def model_slug(name: str) -> str:
    return name.split("/")[-1]


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
        "--enforce-eager",
    ]

    log(f"{'=' * 60}")
    log(f"  Launching: {model_cfg['name']}")
    log(f"  GPUs: {gpu_str} (TP={tp})")
    log(f"  Port: {VLLM_PORT}")
    log(f"{'=' * 60}")

    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    vllm_log = log_dir / f"vllm_sampling_validity_{model_slug(model_cfg['name'])}.log"
    log_fh = open(vllm_log, "w")

    return subprocess.Popen(
        cmd, env=env,
        stdout=log_fh, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
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
#  Core Logic: Sampling + Log-Prob
# ---------------------------------------------------------------

def strip_thinking(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def sample_responses(
    model_name: str, prompt_text: str, n: int = NUM_SAMPLES,
    port: int = VLLM_PORT, max_retries: int = 3,
) -> list[str]:
    """Sample n responses at temperature=1 via chat completions."""
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": GENERATION_MAX_TOKENS,
        "temperature": SAMPLING_TEMPERATURE,
        "n": n,
    }

    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=payload, timeout=600)
            r.raise_for_status()
            choices = r.json()["choices"]
            results = []
            for c in choices:
                text = strip_thinking(c["message"]["content"])
                if text:
                    results.append(text)
            if not results:
                raise ValueError("All responses empty after stripping")
            return results
        except Exception as e:
            if attempt < max_retries - 1:
                log(f"    Sample retry {attempt + 1}/{max_retries}: {e}")
                time.sleep(5)
            else:
                log(f"    Sample FAILED after {max_retries} attempts: {e}")
                return []


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

            resp_logprobs = all_logprobs[prompt_len:-1]

            valid = [v for v in resp_logprobs if v is not None]
            total = sum(valid)
            n = len(valid)

            return {
                "total_logprob": round(total, 6),
                "num_tokens": n,
                "normalized_logprob": round(total / n, 6) if n > 0 else None,
            }
        except Exception as e:
            if attempt < max_retries - 1:
                log(f"      LP retry {attempt + 1}/{max_retries}: {e}")
                time.sleep(5)
            else:
                log(f"      LP FAILED: {e}")
                return {
                    "total_logprob": None,
                    "num_tokens": 0,
                    "normalized_logprob": None,
                    "error": str(e),
                }


# ---------------------------------------------------------------
#  Analysis Helpers
# ---------------------------------------------------------------

def percentile_among(value: float, reference: list[float]) -> float:
    """Fraction of reference values that are <= value (0–100 scale)."""
    if not reference:
        return None
    count = sum(1 for v in reference if v <= value)
    return round(100.0 * count / len(reference), 2)


def z_score(value: float, reference: list[float]) -> float | None:
    if len(reference) < 2:
        return None
    mean = sum(reference) / len(reference)
    std = math.sqrt(sum((v - mean) ** 2 for v in reference) / (len(reference) - 1))
    if std == 0:
        return 0.0
    return round((value - mean) / std, 4)


def primary_construct(correlations: list) -> str:
    if not correlations:
        return "Unknown"
    return max(correlations, key=lambda x: x[1])[0]


# ---------------------------------------------------------------
#  I/O
# ---------------------------------------------------------------

def _result_path(model_name: str) -> Path:
    return OUT_DIR / f"{model_slug(model_name)}.json"


def save_result(result: dict, model_name: str):
    path = _result_path(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log(f"  => Saved: {path.relative_to(ROOT)}")


def load_existing(model_name: str) -> dict | None:
    path = _result_path(model_name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_rq1_results(model_name: str) -> dict | None:
    path = RQ1_DIR / f"{model_slug(model_name)}.json"
    if not path.exists():
        log(f"  WARNING: No RQ1 results at {path}")
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------
#  Summary
# ---------------------------------------------------------------

def compute_summary(results: list) -> dict:
    n = len(results)
    if n == 0:
        return {"num_scenarios": 0}

    pctiles_norm = []
    pctiles_total = []
    z_scores_norm = []
    z_scores_total = []
    coverage_norm = []
    coverage_total = []
    vp_rank1_pctiles_norm = []
    vp_rank1_pctiles_total = []

    for r in results:
        vp_analysis = r.get("vp_analysis", [])
        sampled = r.get("sampled_logprobs", {})
        sampled_norm = sampled.get("normalized_logprobs", [])
        sampled_total = sampled.get("total_logprobs", [])

        valid_sampled_norm = [v for v in sampled_norm if v is not None]
        valid_sampled_total = [v for v in sampled_total if v is not None]

        for va in vp_analysis:
            p = va.get("percentile_normalized")
            if p is not None:
                pctiles_norm.append(p)
            p = va.get("percentile_total")
            if p is not None:
                pctiles_total.append(p)

            z = va.get("z_score_normalized")
            if z is not None:
                z_scores_norm.append(z)
            z = va.get("z_score_total")
            if z is not None:
                z_scores_total.append(z)

            nlp = va.get("normalized_logprob")
            if nlp is not None and valid_sampled_norm:
                in_range = min(valid_sampled_norm) <= nlp <= max(valid_sampled_norm)
                coverage_norm.append(1 if in_range else 0)

            tlp = va.get("total_logprob")
            if tlp is not None and valid_sampled_total:
                in_range = min(valid_sampled_total) <= tlp <= max(valid_sampled_total)
                coverage_total.append(1 if in_range else 0)

        rp = r.get("vp_rank1_percentile_normalized")
        if rp is not None:
            vp_rank1_pctiles_norm.append(rp)
        rp = r.get("vp_rank1_percentile_total")
        if rp is not None:
            vp_rank1_pctiles_total.append(rp)

    def safe_mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    return {
        "num_scenarios": n,
        "mean_vp_percentile_normalized": safe_mean(pctiles_norm),
        "mean_vp_percentile_total": safe_mean(pctiles_total),
        "mean_vp_rank1_percentile_normalized": safe_mean(vp_rank1_pctiles_norm),
        "mean_vp_rank1_percentile_total": safe_mean(vp_rank1_pctiles_total),
        "mean_vp_z_score_normalized": safe_mean(z_scores_norm),
        "mean_vp_z_score_total": safe_mean(z_scores_total),
        "vp_coverage_rate_normalized": safe_mean(coverage_norm),
        "vp_coverage_rate_total": safe_mean(coverage_total),
    }


# ---------------------------------------------------------------
#  Main
# ---------------------------------------------------------------

def main():
    log("=" * 60)
    log("  Sampling Log-Probability Validity")
    log("=" * 60)

    free_gpus = get_free_gpus()
    log(f"Free GPUs: {free_gpus}")
    if not free_gpus:
        log("ERROR: No free GPUs. Exiting.")
        sys.exit(1)

    scenarios = json.loads(VP_SURVEY.read_text())
    advice_tpl = PROMPT_ADVICE.read_text().strip()
    chat_tpl = PROMPT_CHAT.read_text().strip()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_summaries: dict = {}

    for model_cfg in MODELS:
        model_name = model_cfg["name"]
        slug = model_slug(model_name)
        tp = model_cfg.get("tensor_parallel", 1)

        log(f"\n{'─' * 50}")
        log(f"  Model: {model_name}")
        log(f"{'─' * 50}")

        rq1_data = load_rq1_results(model_name)
        if rq1_data is None:
            log(f"  SKIP: no RQ1 log-prob results")
            continue

        rq1_map: dict[int, list] = {}
        for r in rq1_data.get("results", []):
            rq1_map[r["portrait_id"]] = r["outputs"]

        free_gpus = get_free_gpus()
        if len(free_gpus) < tp:
            log(f"  SKIP: need {tp} GPU(s), only {len(free_gpus)} free")
            continue

        existing = load_existing(model_name)
        done_pids: set[int] = set()
        if existing:
            done_pids = {r["portrait_id"] for r in existing.get("results", [])}
            log(f"  Resuming: {len(done_pids)}/{len(scenarios)} scenarios already done")

        remaining = [s for s in scenarios if s["portrait_id"] not in done_pids]
        if not remaining:
            log(f"  All {len(scenarios)} scenarios already done")
            summary = compute_summary(existing["results"])
            existing["summary"] = summary
            save_result(existing, model_name)
            all_summaries[slug] = summary
            _log_model_summary(model_name, summary, len(scenarios))
            continue

        result: dict = existing or {
            "model": model_name,
            "survey": "VP",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_scenarios": len(scenarios),
            "num_samples": NUM_SAMPLES,
            "sampling_temperature": SAMPLING_TEMPERATURE,
            "results": [],
        }

        log(f"  Loading tokenizer for {model_name} ...")
        tokenizer = _load_tokenizer(model_name)

        proc = None
        for attempt in range(SERVER_LAUNCH_RETRIES):
            try:
                proc = launch_with_retry(model_cfg)

                consecutive_fails = 0
                new_count = 0

                for i, scenario in enumerate(scenarios):
                    pid = scenario["portrait_id"]
                    if pid in done_pids:
                        continue

                    ptype = get_prompt_type(pid)
                    prompt_text = format_scenario_prompt(
                        scenario["content"], pid, advice_tpl, chat_tpl,
                    )

                    log(f"  [{i + 1}/{len(scenarios)}] pid={pid} ({ptype})")

                    rq1_outs = rq1_map.get(pid)
                    if rq1_outs is None:
                        log(f"    WARNING: no RQ1 data for pid={pid}, skipping")
                        continue

                    # --- Step 1: Sample K responses at temperature=1 ---
                    samples = sample_responses(model_name, prompt_text)
                    if not samples:
                        consecutive_fails += 1
                        if consecutive_fails >= CONSECUTIVE_FAIL_LIMIT:
                            save_result(result, model_name)
                            raise ServerCrashError(
                                f"Server unresponsive ({consecutive_fails} consecutive failures)"
                            )
                        continue
                    consecutive_fails = 0

                    log(f"    Sampled {len(samples)} responses")

                    # --- Step 2: Compute log-prob for each sampled response ---
                    sampled_lps: list[dict] = []
                    sample_failed = False

                    for si, sample_text in enumerate(samples):
                        lp_data = compute_response_logprob(
                            model_name, prompt_text, sample_text, tokenizer,
                        )

                        if lp_data.get("error"):
                            consecutive_fails += 1
                            if consecutive_fails >= CONSECUTIVE_FAIL_LIMIT:
                                save_result(result, model_name)
                                raise ServerCrashError(
                                    f"Server unresponsive ({consecutive_fails} consecutive failures)"
                                )
                            sample_failed = True
                            break
                        consecutive_fails = 0

                        sampled_lps.append({
                            "sample_idx": si,
                            "text_length": len(sample_text),
                            "total_logprob": lp_data["total_logprob"],
                            "num_tokens": lp_data["num_tokens"],
                            "normalized_logprob": lp_data["normalized_logprob"],
                        })

                    if sample_failed:
                        continue

                    sampled_norm_vals = [
                        s["normalized_logprob"] for s in sampled_lps
                        if s["normalized_logprob"] is not None
                    ]
                    sampled_total_vals = [
                        s["total_logprob"] for s in sampled_lps
                        if s["total_logprob"] is not None
                    ]

                    if sampled_total_vals:
                        log(f"    Sampled total lp: "
                            f"mean={sum(sampled_total_vals) / len(sampled_total_vals):.2f} "
                            f"range=[{min(sampled_total_vals):.2f}, {max(sampled_total_vals):.2f}]")
                    else:
                        log("    Sampled total lp: N/A")

                    # --- Step 3: Analyze VP candidates against sampled distribution ---
                    vp_out_map = {o["id"]: o for o in scenario["outputs"]}
                    vp_analysis: list[dict] = []

                    for rq1_out in rq1_outs:
                        oid = rq1_out["output_id"]
                        vp_norm = rq1_out.get("normalized_logprob")
                        vp_total = rq1_out.get("total_logprob")

                        construct = "Unknown"
                        if oid in vp_out_map:
                            construct = primary_construct(
                                vp_out_map[oid]["correlations"]
                            )

                        pctile_norm = (
                            percentile_among(vp_norm, sampled_norm_vals)
                            if vp_norm is not None and sampled_norm_vals else None
                        )
                        pctile_total = (
                            percentile_among(vp_total, sampled_total_vals)
                            if vp_total is not None and sampled_total_vals else None
                        )
                        z_norm = (
                            z_score(vp_norm, sampled_norm_vals)
                            if vp_norm is not None and sampled_norm_vals else None
                        )
                        z_total = (
                            z_score(vp_total, sampled_total_vals)
                            if vp_total is not None and sampled_total_vals else None
                        )

                        vp_analysis.append({
                            "output_id": oid,
                            "primary_construct": construct,
                            "normalized_logprob": vp_norm,
                            "total_logprob": vp_total,
                            "percentile_normalized": pctile_norm,
                            "percentile_total": pctile_total,
                            "z_score_normalized": z_norm,
                            "z_score_total": z_total,
                        })

                        log(f"    VP out {oid} ({construct}): "
                            f"total_lp={f'{vp_total:.2f}' if vp_total is not None else 'N/A'} "
                            f"pctile={pctile_total}% "
                            f"z={z_total}")

                    # VP rank-1 percentile
                    vp_sorted_norm = sorted(
                        [a for a in vp_analysis if a["normalized_logprob"] is not None],
                        key=lambda a: a["normalized_logprob"], reverse=True,
                    )
                    vp_sorted_total = sorted(
                        [a for a in vp_analysis if a["total_logprob"] is not None],
                        key=lambda a: a["total_logprob"], reverse=True,
                    )

                    vp_r1_pctile_norm = (
                        vp_sorted_norm[0]["percentile_normalized"]
                        if vp_sorted_norm else None
                    )
                    vp_r1_pctile_total = (
                        vp_sorted_total[0]["percentile_total"]
                        if vp_sorted_total else None
                    )

                    entry = {
                        "portrait_id": pid,
                        "prompt_type": ptype,
                        "sampled_responses": [
                            {
                                "sample_idx": s["sample_idx"],
                                "text_length": s["text_length"],
                                "total_logprob": s["total_logprob"],
                                "num_tokens": s["num_tokens"],
                                "normalized_logprob": s["normalized_logprob"],
                            }
                            for s in sampled_lps
                        ],
                        "sampled_logprobs": {
                            "normalized_logprobs": sampled_norm_vals,
                            "total_logprobs": sampled_total_vals,
                            "mean_normalized": (
                                round(sum(sampled_norm_vals) / len(sampled_norm_vals), 6)
                                if sampled_norm_vals else None
                            ),
                            "std_normalized": (
                                round(math.sqrt(
                                    sum((v - sum(sampled_norm_vals) / len(sampled_norm_vals)) ** 2
                                        for v in sampled_norm_vals)
                                    / (len(sampled_norm_vals) - 1)
                                ), 6)
                                if len(sampled_norm_vals) > 1 else None
                            ),
                            "mean_total": (
                                round(sum(sampled_total_vals) / len(sampled_total_vals), 6)
                                if sampled_total_vals else None
                            ),
                            "std_total": (
                                round(math.sqrt(
                                    sum((v - sum(sampled_total_vals) / len(sampled_total_vals)) ** 2
                                        for v in sampled_total_vals)
                                    / (len(sampled_total_vals) - 1)
                                ), 6)
                                if len(sampled_total_vals) > 1 else None
                            ),
                        },
                        "vp_analysis": vp_analysis,
                        "vp_rank1_percentile_normalized": vp_r1_pctile_norm,
                        "vp_rank1_percentile_total": vp_r1_pctile_total,
                    }

                    result["results"].append(entry)
                    done_pids.add(pid)
                    new_count += 1

                    log(f"    VP rank-1 percentile (total): {vp_r1_pctile_total}%")

                    if new_count % 5 == 0:
                        save_result(result, model_name)

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

        save_result(result, model_name)
        summary = compute_summary(result["results"])
        result["summary"] = summary
        save_result(result, model_name)
        all_summaries[slug] = summary

        _log_model_summary(model_name, summary, len(scenarios))

    # ===========================================================
    #  Cross-model summary
    # ===========================================================
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False))

    log(f"\n{'=' * 60}")
    log("  Final Summary: Sampling Log-Prob Validity")
    log(f"{'=' * 60}")
    log(f"  {'Model':<45s} {'VP-Pct(T)':>10s} {'R1-Pct(T)':>10s} "
        f"{'Z(T)':>7s} {'Cov(T)':>7s}")
    log(f"  {'─' * 45} {'─' * 10} {'─' * 10} {'─' * 7} {'─' * 7}")
    for slug, s in all_summaries.items():
        vp_pct = s.get("mean_vp_percentile_total")
        r1_pct = s.get("mean_vp_rank1_percentile_total")
        z_t = s.get("mean_vp_z_score_total")
        cov = s.get("vp_coverage_rate_total")
        log(f"  {slug:<45s} "
            f"{f'{vp_pct:.1f}%' if vp_pct is not None else 'N/A':>10s} "
            f"{f'{r1_pct:.1f}%' if r1_pct is not None else 'N/A':>10s} "
            f"{f'{z_t:.3f}' if z_t is not None else '  N/A':>7s} "
            f"{f'{cov:.1%}' if cov is not None else '  N/A':>7s}")
    log(f"{'=' * 60}")
    log("  All done!")
    log(f"{'=' * 60}")


def _log_model_summary(model_name: str, summary: dict, total: int):
    log(f"\n  === {model_name} Summary ===")
    log(f"  Scenarios:                    {summary['num_scenarios']}/{total}")

    vp_pct = summary.get("mean_vp_percentile_total")
    r1_pct = summary.get("mean_vp_rank1_percentile_total")
    z_t = summary.get("mean_vp_z_score_total")
    cov_t = summary.get("vp_coverage_rate_total")

    log(f"  Mean VP percentile (total):   {f'{vp_pct:.1f}%' if vp_pct is not None else 'N/A'}")
    log(f"  VP rank-1 percentile (total): {f'{r1_pct:.1f}%' if r1_pct is not None else 'N/A'}")
    log(f"  Mean VP z-score (total):      {f'{z_t:.3f}' if z_t is not None else 'N/A'}")
    log(f"  VP coverage rate (total):     {f'{cov_t:.1%}' if cov_t is not None else 'N/A'}")


if __name__ == "__main__":
    main()
