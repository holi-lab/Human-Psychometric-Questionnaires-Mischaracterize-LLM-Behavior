#!/usr/bin/env python3
"""
Item-construct recognition on cue-reduced items (GPU).

Copied/adapted from RQ3/item_construct_recognition.py. Differences:
  - Only PVQ + BFI44, pointed at surveys_cueless/
  - Output to RQ3/cue_reduction/results/recognition_cueless/{survey}/{model}.json
  - vLLM port 8011
  - 8 models: six TP1 models plus two large TP4 models
  - Uses the ORIGINAL recognition prompt templates (prompts/item_construct_recognition_{bfi,pvq}.txt)

If cue removal works, recognition F1 should drop sharply vs. the original items.

Run with enough free GPUs for the selected models:
    python RQ3/cue_reduction/scripts/run_recognition_cueless.py
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

CUE_DIR = Path(__file__).resolve().parent.parent
ROOT = CUE_DIR.parent.parent  # Official/ repo root
CUELESS_DIR = CUE_DIR / "surveys_cueless"
OUT_ROOT = CUE_DIR / "results" / "recognition_cueless"

VLLM_PORT = 8011
GPU_MEMORY_UTILIZATION = 0.90
DTYPE = "bfloat16"
SERVER_LAUNCH_RETRIES = 3
SERVER_CRASH_RESTARTS = 10
CONSECUTIVE_FAIL_LIMIT = 5
MAX_TOKENS = 1024
PARSE_RETRIES = 3

PVQ_DEFS = {
    "Power": "Social status and prestige, control or dominance over people and resources",
    "Achievement": "Personal success through demonstrating competence according to social standards",
    "Hedonism": "Pleasure and sensuous gratification for oneself",
    "Stimulation": "Excitement, novelty, and challenge in life",
    "Self-Direction": "Independent thought and action - choosing, creating, exploring",
    "Universalism": "Understanding, appreciation, tolerance, and protection for the welfare of all people and for nature",
    "Benevolence": "Preserving and enhancing the welfare of people with whom one is in frequent personal contact",
    "Tradition": "Respect, commitment, and acceptance of the customs and ideas that one's culture or religion provides",
    "Conformity": "Restraint of actions, inclinations, and impulses likely to upset or harm others and violate social expectations or norms",
    "Security": "Safety, harmony, and stability of society, of relationships, and of self",
}
BFI_DEFS = {
    "Openness": "The tendency to be imaginative, curious, and receptive to new ideas, experiences, and perspectives",
    "Conscientiousness": "The tendency to be organized, responsible, hardworking, and goal-directed",
    "Extraversion": "The tendency to be sociable, assertive, active, and to experience positive emotions",
    "Agreeableness": "The tendency to be compassionate, cooperative, trusting, and helpful toward others",
    "Neuroticism": "The tendency to experience negative emotions such as anxiety, depression, and emotional instability",
}

SURVEYS = {
    "PVQ": {"survey_file": CUELESS_DIR / "PVQ.json", "construct_key": "value",
            "construct_definitions": PVQ_DEFS, "prompt": "pvq"},
    "BFI44": {"survey_file": CUELESS_DIR / "BFI44.json", "construct_key": "trait",
              "construct_definitions": BFI_DEFS, "prompt": "bfi"},
}

# Keep the six smaller models at TP1 to avoid the vLLM TP2 engine-init issue.
# The two large Qwen checkpoints use the known-good TP4 settings from RQ3.
MODELS = [
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

PROMPT_TEMPLATES = {
    "bfi": ROOT / "prompts" / "item_construct_recognition_bfi.txt",
    "pvq": ROOT / "prompts" / "item_construct_recognition_pvq.txt",
}

_THINK_RE = None
_active_proc = None


class ServerCrashError(Exception):
    pass


def log(msg=""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def model_slug(name):
    return name.split("/")[-1]


def get_free_gpus(threshold_mb=1000):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"], text=True)
        free = []
        for line in out.strip().splitlines():
            idx, mem = line.split(",")
            if int(mem.strip()) < threshold_mb:
                free.append(int(idx.strip()))
        return free
    except Exception:
        return []


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


def launch_vllm(model_cfg):
    ensure_port_free(VLLM_PORT)
    tp = model_cfg.get("tensor_parallel", 1)
    slurm_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    slurm_devices = [d for d in slurm_cvd.split(",") if d.strip() != ""]
    if slurm_devices:
        if len(slurm_devices) < tp:
            raise RuntimeError(f"Need {tp} GPU(s), job allocation gave {slurm_devices}")
        gpu_str = ",".join(slurm_devices[:tp])
    else:
        free_gpus = get_free_gpus()
        if len(free_gpus) < tp:
            raise RuntimeError(f"Need {tp} GPU(s), have {free_gpus}")
        gpu_str = ",".join(str(g) for g in free_gpus[:tp])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_str
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["VLLM_LOGGING_LEVEL"] = "WARNING"
    gmu = model_cfg.get("gpu_mem_util", GPU_MEMORY_UTILIZATION)
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_cfg["name"], "--host", "0.0.0.0", "--port", str(VLLM_PORT),
        "--tensor-parallel-size", str(tp),
        "--max-model-len", str(model_cfg.get("max_model_len", 2048)),
        "--gpu-memory-utilization", str(gmu),
        "--dtype", DTYPE, "--trust-remote-code",
        "--enable-prefix-caching", "--max-num-seqs", "8",
    ]
    if model_cfg.get("enforce_eager", False):
        cmd.append("--enforce-eager")
    log("=" * 60)
    log(f"  Launching: {model_cfg['name']}  GPUs: {gpu_str} (TP={tp})  Port: {VLLM_PORT}")
    log("=" * 60)
    log_dir = CUE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_dir / f"vllm_rec_{model_slug(model_cfg['name'])}.log", "w")
    return subprocess.Popen(cmd, env=env, stdout=log_fh,
                            stderr=subprocess.STDOUT, preexec_fn=os.setsid)


def wait_for_server(proc, port, timeout=3600, interval=10):
    url = f"http://localhost:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited code {proc.returncode} during startup")
        try:
            if requests.get(url, timeout=5).status_code == 200:
                log(f"  Server ready on port {port}")
                return
        except requests.RequestException:
            pass
        time.sleep(interval)
    raise TimeoutError(f"server not ready after {timeout}s")


def kill_server(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


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
                time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"Failed to launch {model_cfg['name']}")


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


def _check_server_alive():
    if _active_proc is not None and _active_proc.poll() is not None:
        raise ServerCrashError(f"vLLM exited (code={_active_proc.returncode})")


def query_model(model_name, messages, port=VLLM_PORT, max_retries=5,
                temperature=0.0, max_tokens=MAX_TOKENS):
    url = f"http://localhost:{port}/v1/chat/completions"
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    temps = [temperature] + ([0.3, 0.5, 0.7, 1.0] if temperature == 0.0 else [])
    for temp in temps:
        payload = {"model": model_name, "messages": messages,
                   "max_tokens": max_tokens, "temperature": temp}
        for attempt in range(max_retries):
            _check_server_alive()
            try:
                r = requests.post(url, json=payload, timeout=120)
                r.raise_for_status()
                msg = r.json()["choices"][0]["message"]
                raw = msg.get("content")
                if raw is None:
                    raise ValueError("null content")
                content, reasoning = _extract_reasoning(raw, msg)
                return {"content": content, "reasoning": reasoning}
            except ServerCrashError:
                raise
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    if temp != temps[-1]:
                        break
                    raise


def parse_yes_no(response):
    resp = response.strip().lower()
    if resp.startswith("yes"):
        return True
    if resp.startswith("no"):
        return False
    if "yes" in resp:
        return True
    if "no" in resp:
        return False
    return None


def query_and_parse_binary(model_name, prompt_text, port=VLLM_PORT):
    reasoning, raw_response = None, ""
    for attempt in range(1 + PARSE_RETRIES):
        resp = query_model(model_name, prompt_text, port=port,
                           temperature=0.0, max_tokens=MAX_TOKENS)
        raw_response = resp["content"]
        reasoning = resp["reasoning"]
        parsed = parse_yes_no(raw_response)
        if parsed is not None:
            return raw_response, parsed, reasoning
    return raw_response, None, reasoning


def result_path(survey_name, model_name):
    return OUT_ROOT / survey_name / f"{model_slug(model_name)}.json"


def load_existing(survey_name, model_name):
    p = result_path(survey_name, model_name)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def save_result(survey_name, model_name, result):
    p = result_path(survey_name, model_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, ensure_ascii=False))


def build_items(survey_name):
    cfg = SURVEYS[survey_name]
    data = json.loads(cfg["survey_file"].read_text())
    template = PROMPT_TEMPLATES[cfg["prompt"]].read_text().strip()
    items = []
    for item_id in sorted(data, key=int):
        d = data[item_id]
        items.append({
            "item_key": f"item_{item_id}",
            "question_text": d.get("en_neutral", "").strip(),
            "ground_truth_construct": d["meta_data"][cfg["construct_key"]],
        })
    return items, template, cfg["construct_definitions"]


def prediction_complete(pred):
    return bool(pred) and pred.get("raw_response") not in ("", None) and pred.get("parsed") is not None


def item_needs_repair(item_result, construct_names):
    if not item_result:
        return True
    preds = item_result.get("predictions", {})
    return any(not prediction_complete(preds.get(c)) for c in construct_names)


def count_complete(items, results, construct_names):
    return sum(1 for it in items if not item_needs_repair(results.get(it["item_key"]), construct_names))


def run_survey(model_name, survey_name):
    items, template, defs = build_items(survey_name)
    construct_names = list(defs.keys())
    existing = load_existing(survey_name, model_name)
    if existing is None:
        result = {"model": model_name, "survey": survey_name,
                  "timestamp": datetime.now(timezone.utc).isoformat(),
                  "cue_removed": True,
                  "construct_definitions_used": dict(defs), "results": {}}
    else:
        result = existing
        result.setdefault("results", {})
        result.setdefault("construct_definitions_used", dict(defs))

    done = count_complete(items, result["results"], construct_names)
    log(f"  Resuming {survey_name}: {done}/{len(items)} items complete")
    consecutive_fails = 0

    for idx, item in enumerate(items, 1):
        existing_item = result["results"].get(item["item_key"])
        if not item_needs_repair(existing_item, construct_names):
            continue
        predictions = dict(existing_item.get("predictions", {})) if existing_item else {}
        for cname, cdef in defs.items():
            if prediction_complete(predictions.get(cname)):
                continue
            prompt_text = template.format(
                question_text=item["question_text"],
                construct_name=cname, construct_definition=cdef)
            try:
                raw, parsed, reasoning = query_and_parse_binary(model_name, prompt_text)
                consecutive_fails = 0
            except ServerCrashError:
                save_result(survey_name, model_name, result)
                raise
            except Exception as e:
                consecutive_fails += 1
                log(f"    [{survey_name}] {item['item_key']}/{cname} ERR: {e}")
                if consecutive_fails >= CONSECUTIVE_FAIL_LIMIT:
                    save_result(survey_name, model_name, result)
                    raise ServerCrashError("too many consecutive failures")
                raw, parsed, reasoning = "", None, None
            predictions[cname] = {"raw_response": raw, "parsed": parsed, "reasoning": reasoning}
        result["results"][item["item_key"]] = {
            "question_text": item["question_text"],
            "ground_truth_construct": item["ground_truth_construct"],
            "predictions": predictions,
        }
        save_result(survey_name, model_name, result)
        log(f"    [{survey_name}] {idx}/{len(items)} {item['item_key']} saved")


def server_healthy(port=VLLM_PORT):
    try:
        return requests.get(f"http://localhost:{port}/v1/models", timeout=10).status_code == 200
    except Exception:
        return False


def main():
    global _active_proc
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default=None,
                    help="comma-separated model slugs or full names to include")
    args = ap.parse_args()

    models = MODELS
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        models = [m for m in MODELS
                  if m["name"] in wanted or model_slug(m["name"]) in wanted]

    log("=" * 60)
    log("  Item-construct recognition on cue-reduced items")
    log(f"  Models: {[model_slug(m['name']) for m in models]}")
    log("=" * 60)
    free = get_free_gpus()
    log(f"Free GPUs: {free}")
    if not free:
        log("ERROR: No free GPUs.")
        sys.exit(1)

    for model_cfg in models:
        model_name = model_cfg["name"]
        tp = model_cfg.get("tensor_parallel", 1)
        if len(get_free_gpus()) < tp:
            log(f"SKIP {model_name}: need {tp} GPUs")
            continue

        # skip if both surveys already complete
        pending = []
        for sn in SURVEYS:
            items, _, defs = build_items(sn)
            ex = load_existing(sn, model_name)
            res = ex.get("results", {}) if ex else {}
            if count_complete(items, res, list(defs)) < len(items):
                pending.append(sn)
        if not pending:
            log(f"SKIP {model_name}: all surveys complete")
            continue

        log("─" * 50)
        log(f"  Model: {model_name}  Pending: {pending}")
        log("─" * 50)

        proc = None
        crash_count = 0
        try:
            proc = launch_with_retry(model_cfg)
            _active_proc = proc
            i = 0
            while i < len(pending):
                sn = pending[i]
                try:
                    run_survey(model_name, sn)
                    i += 1
                except ServerCrashError as e:
                    crash_count += 1
                    log(f"  SERVER CRASH during {sn}: {e}")
                    if proc:
                        kill_server(proc)
                        proc = None
                    _active_proc = None
                    if crash_count >= SERVER_CRASH_RESTARTS:
                        log(f"  Too many crashes; giving up on {model_name}")
                        break
                    time.sleep(min(30 * crash_count, 120))
                    proc = launch_with_retry(model_cfg)
                    _active_proc = proc
        except Exception as e:
            log(f"ERROR with {model_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if proc:
                kill_server(proc)
            _active_proc = None

    log("=" * 60)
    log("  All done!")
    log("=" * 60)


if __name__ == "__main__":
    main()
