#!/usr/bin/env python3
"""
Cue-reduced 2x2 runner: score base/instruct checkpoints on cue-reduced items.

One vLLM server per model, three tasks, all via LOG-PROBABILITIES so that
base (non-chat) models can be scored the same way as instruct models:

  task=established : PVQ-40 + BFI-44 constrained Likert scoring.
      Prompt lists numbered options (1..6 PVQ / 1..5 BFI); we read the
      logprobs of the digit tokens at the answer position and take BOTH
      argmax and probability-weighted mean.  normal + reversed option
      order, mirroring RQ1/established.py.  Output schema is compatible
      with RQ2/aic_established.py.

  task=ecological  : VP generation-probability profile.
      log P(response | scenario) token-by-token, same as RQ1/ecological.py,
      but with a --format switch:  chat  -> tokenizer.apply_chat_template
                                   plain -> scenario + "\n\n" + response.

  task=recognition : item-construct recognition (RQ3), constrained yes/no.
      Reads logprobs of yes/no tokens at the answer position.

Format handling:
  --format plain : NO chat template. Used for base models (they have none).
  --format chat  : tokenizer.apply_chat_template. Used for instruct models
                   in their native format.
  For the anchor comparison we also run instruct models with --format plain
  so base-vs-instruct is not confounded by prompt format.

Usage:
    python RQ3/cueless_2x2/run.py --model Qwen/Qwen2.5-7B --format plain \
        --task all --surveys PVQ,BFI44 --out-tag base --limit 0

Results (under this dir):
    results/{tag}/established/{SURVEY}/{model_slug}.json
    results/{tag}/ecological/{model_slug}.json
    results/{tag}/recognition/{SURVEY}/{model_slug}.json
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

# Official/ root (two levels up from RQ3/cueless_2x2/)
OFFICIAL = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
# Established items are loaded from this directory (default: the cue-reduced
# rewrites). Override with --survey-dir. Same JSON schema as surveys/.
SURVEY_DIR = OFFICIAL / "RQ3" / "cue_reduction" / "surveys_cueless"

VLLM_PORT = 8012
GPU_MEMORY_UTILIZATION = 0.85
DTYPE = "bfloat16"


def log(msg: str = ""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def model_slug(name: str) -> str:
    return name.split("/")[-1]


# ===========================================================
#  Survey configs
# ===========================================================
PVQ_OPTIONS = [  # score value, label   (score 1..6)
    (1, "Not like me at all"),
    (2, "Not like me"),
    (3, "A little like me"),
    (4, "Somewhat like me"),
    (5, "Like me"),
    (6, "Very much like me"),
]
BFI_OPTIONS = [
    (1, "Disagree strongly"),
    (2, "Disagree a little"),
    (3, "Neither agree nor disagree"),
    (4, "Agree a little"),
    (5, "Agree strongly"),
]

ESTABLISHED = {
    "PVQ": {"file": "surveys/PVQ.json", "type": "pvq", "construct_key": "value",
            "scale_max": 6, "options": PVQ_OPTIONS},
    "PVQ21": {"file": "surveys/PVQ21.json", "type": "pvq", "construct_key": "value",
              "scale_max": 6, "options": PVQ_OPTIONS},
    "BFI44": {"file": "surveys/BFI44.json", "type": "bfi", "construct_key": "trait",
              "scale_max": 5, "options": BFI_OPTIONS},
    "BFI10": {"file": "surveys/BFI10.json", "type": "bfi", "construct_key": "trait",
              "scale_max": 5, "options": BFI_OPTIONS},
}

RECOG_DEFS = {
    "pvq": {
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
    },
    "bfi": {
        "Openness": "The tendency to be imaginative, curious, and receptive to new ideas, experiences, and perspectives",
        "Conscientiousness": "The tendency to be organized, responsible, hardworking, and goal-directed",
        "Extraversion": "The tendency to be sociable, assertive, active, and to experience positive emotions",
        "Agreeableness": "The tendency to be compassionate, cooperative, trusting, and helpful toward others",
        "Neuroticism": "The tendency to experience negative emotions such as anxiety, depression, and emotional instability",
    },
}
RECOG_DEFS["vp"] = {**RECOG_DEFS["pvq"], **RECOG_DEFS["bfi"]}

RECOG_SURVEY_CFG = {
    "PVQ": ("pvq", "value", "surveys/PVQ.json"),
    "PVQ21": ("pvq", "value", "surveys/PVQ21.json"),
    "BFI44": ("bfi", "trait", "surveys/BFI44.json"),
    "BFI10": ("bfi", "trait", "surveys/BFI10.json"),
    "VP": ("vp", None, "surveys/VP.json"),
}
DISPLAY_NAME_MAP = {
    "Self_Direction": "Self-Direction", "Self_Enhancement": "Self-Enhancement",
    "Self_Transcendence": "Self-Transcendence", "Openness_to_Change": "Openness to Change",
}
VP_POSITIVE_THRESHOLD = 0.3
CORR_THRESHOLD = 0.3


# ===========================================================
#  vLLM server management
# ===========================================================
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


def wait_for_gpu_headroom(tp, min_free_mb=70000, timeout=180):
    """Block until at least *tp* GPUs each have >= min_free_mb free, so a prior
    vLLM EngineCore that is still releasing memory does not fail the next launch.
    Counts free memory across the GPUs visible to nvidia-smi."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits"], text=True)
            free = [int(x.strip()) for x in out.strip().splitlines()]
            if sum(1 for f in free if f >= min_free_mb) >= tp:
                return
        except Exception:
            pass
        log(f"  Waiting for GPU headroom ({tp} GPUs >= {min_free_mb}MB free) ...")
        time.sleep(10)
    log("  WARNING: GPU headroom wait timed out; launching anyway.")


def launch_vllm(model_name, tp, max_model_len, enforce_eager):
    ensure_port_free(VLLM_PORT)
    wait_for_gpu_headroom(tp)
    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", model_name, "--host", "0.0.0.0", "--port", str(VLLM_PORT),
           "--tensor-parallel-size", str(tp),
           "--max-model-len", str(max_model_len),
           "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
           "--dtype", DTYPE, "--trust-remote-code",
           "--enable-prefix-caching", "--max-logprobs", "40"]
    if enforce_eager:
        cmd.append("--enforce-eager")
    log(f"Launching {model_name} (TP={tp}, eager={enforce_eager}) on port {VLLM_PORT}")
    logf = open(HERE / "logs" / f"vllm_{model_slug(model_name)}.log", "w")
    return subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)


def wait_for_server(proc, timeout=3600, interval=10):
    url = f"http://localhost:{VLLM_PORT}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited code {proc.returncode} during startup")
        try:
            if requests.get(url, timeout=5).status_code == 200:
                log(f"Server ready on port {VLLM_PORT}")
                return
        except requests.RequestException:
            pass
        time.sleep(interval)
    raise TimeoutError("server not ready")


def kill_server(proc):
    """Terminate the whole vLLM process GROUP (APIServer + EngineCore workers).

    proc.wait() only reaps the tracked APIServer; the EngineCore child holds the
    GPU memory, so we ALWAYS SIGKILL the process group at the end. We cannot kill
    by uid because other jobs may share this Unix user."""
    if proc is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        pgid = None
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        # Force-kill the entire group regardless of graceful-exit outcome.
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


# ===========================================================
#  Low-level logprob calls
# ===========================================================
def _completions(payload, max_retries=4):
    url = f"http://localhost:{VLLM_PORT}/v1/completions"
    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=payload, timeout=300)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                raise


def next_token_logprobs(model_name, prompt, top=40):
    """Return dict token->logprob for the single next token."""
    data = _completions({
        "model": model_name, "prompt": prompt,
        "max_tokens": 1, "temperature": 0.0, "logprobs": top, "echo": False,
    })
    lp = data["choices"][0]["logprobs"]
    tl = lp.get("top_logprobs") or [{}]
    return tl[0] if tl and tl[0] else {}


def echo_response_logprob(model_name, prompt_prefix, response_text):
    """log P(response | prompt) via echo. prompt_prefix already contains any
    separator; response_text is appended raw. Returns dict."""
    full = prompt_prefix + response_text
    data = _completions({
        "model": model_name, "prompt": full,
        "max_tokens": 1, "temperature": 1.0, "logprobs": 1, "echo": True,
    })
    lp = data["choices"][0]["logprobs"]
    all_lps = lp["token_logprobs"]
    all_tokens = lp["tokens"]
    text_offset = lp.get("text_offset")
    # Find split index by character offset of the prefix length.
    prefix_len = len(prompt_prefix)
    if text_offset and len(text_offset) == len(all_tokens):
        split = 0
        for i, off in enumerate(text_offset):
            if off < prefix_len:
                split = i + 1
        # split = number of tokens fully within prefix
    else:
        split = None
    # echo returns one extra generated token at the end -> drop last
    if split is None:
        # fallback: cannot reliably slice
        resp_lps = [v for v in all_lps if v is not None]
        resp_tokens = all_tokens
    else:
        resp_lps = all_lps[split:]
        resp_tokens = all_tokens[split:]
        # drop trailing generated token (max_tokens=1)
        if resp_lps:
            resp_lps = resp_lps[:-1]
            resp_tokens = resp_tokens[:-1]
    valid = [v for v in resp_lps if v is not None]
    total = sum(valid)
    n = len(valid)
    return {
        "total_logprob": round(total, 6),
        "num_tokens": n,
        "normalized_logprob": round(total / n, 6) if n else None,
        "tokens": resp_tokens,
    }


# ===========================================================
#  Digit / yes-no extraction helpers
# ===========================================================
def _prob_for_targets(top_logprobs, targets):
    """targets: list of acceptable string forms (case/space variants).
    Returns best (max) probability mass matched for the target group."""
    best = None
    for tok, lp in top_logprobs.items():
        t = tok.strip().lower()
        if t in targets:
            p = math.exp(lp)
            if best is None or p > best:
                best = p
    return best or 0.0


def score_likert_item(model_name, prompt, options, reverse_presentation):
    """options: list of (score_value, label). Prompt already lists them 1..K.
    reverse_presentation: if True the digit->score mapping is reversed.
    Returns (raw_score_argmax, raw_score_weighted, digit_probs, coverage)."""
    K = len(options)
    top = next_token_logprobs(model_name, prompt, top=40)
    # digit d (1..K) shown at position d; presented label index:
    #   normal:   position d -> option[d-1] (score = option[d-1][0])
    #   reversed: position d -> option[K-d] (score = option[K-d][0])
    digit_prob = {}
    for d in range(1, K + 1):
        digit_prob[d] = _prob_for_targets(top, {str(d)})
    total_mass = sum(digit_prob.values())
    if total_mass <= 0:
        return None, None, digit_prob, 0.0
    # map digit -> score
    def digit_to_score(d):
        if reverse_presentation:
            return options[K - d][0]
        return options[d - 1][0]
    # argmax
    best_d = max(digit_prob, key=lambda d: digit_prob[d])
    raw_argmax = digit_to_score(best_d)
    # weighted mean over normalized digit probs
    wsum = sum(digit_prob[d] * digit_to_score(d) for d in digit_prob)
    raw_weighted = wsum / total_mass
    return raw_argmax, raw_weighted, digit_prob, total_mass


YES_TARGETS = {"yes", "y"}
NO_TARGETS = {"no", "n"}


def score_yes_no(model_name, prompt):
    top = next_token_logprobs(model_name, prompt, top=40)
    p_yes = _prob_for_targets(top, YES_TARGETS)
    p_no = _prob_for_targets(top, NO_TARGETS)
    if p_yes == 0.0 and p_no == 0.0:
        return None, p_yes, p_no
    return (p_yes >= p_no), p_yes, p_no


# ===========================================================
#  Prompt builders
# ===========================================================
def build_likert_prompt(cfg, content, reverse_presentation, fmt, tokenizer):
    options = cfg["options"]
    K = len(options)
    if reverse_presentation:
        shown = [(i + 1, options[K - 1 - i][1]) for i in range(K)]
    else:
        shown = [(i + 1, options[i][1]) for i in range(K)]
    opt_lines = "\n".join(f"{d} = {lab}" for d, lab in shown)

    if cfg["type"] == "pvq":
        body = (
            "Here we briefly describe a person. Please read the description and "
            "think about how much this person is or is not like you.\n\n"
            f"{content}\n\n"
            "How much like you is this person? Answer with a single number "
            f"from 1 to {K}:\n{opt_lines}"
        )
    else:
        body = (
            "Here are a number of characteristics that may or may not apply to "
            "you. Please indicate the extent to which you agree or disagree that "
            "each statement describes you.\n\n"
            f"I see myself as someone who {content}\n\n"
            f"Answer with a single number from 1 to {K}:\n{opt_lines}"
        )
    if fmt == "chat":
        msgs = [{"role": "user", "content": body}]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        return prompt  # model emits the digit next
    else:
        return body + "\n\nAnswer: "


def build_recog_prompt(kind, question_text, cname, cdef, fmt, tokenizer):
    if kind == "bfi":
        body = (f"I see myself as someone who {question_text}\n\n"
                "Does this item measure the following construct, including "
                f"reverse-scored items?\n\n{cname}: {cdef}\n\n"
                'Answer with only "yes" or "no".')
    elif kind == "pvq":
        body = (f"{question_text}\n\nDoes this item measure the following "
                f"construct?\n\n{cname}: {cdef}\n\n"
                'Answer with only "yes" or "no".')
    else:  # vp
        body = (f"Question: {question_text}\n\nDoes this question primarily "
                f"measure the following construct?\n\n{cname}: {cdef}\n\n"
                'Answer with only "yes" or "no".')
    if fmt == "chat":
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": body}],
            tokenize=False, add_generation_prompt=True)
        return prompt
    return body + "\n\nAnswer: "


def format_vp_prefix(scenario_content, pid, advice_tpl, chat_tpl, fmt, tokenizer):
    if 1000 <= pid < 3000:
        user = (advice_tpl.replace("{title}", scenario_content["title"])
                .replace("{text}", scenario_content["text"]))
        ptype = "advice"
    elif 3000 <= pid < 5000:
        user = chat_tpl.replace("{text}", scenario_content["text"])
        ptype = "chat"
    else:
        raise ValueError(pid)
    if fmt == "chat":
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
    else:
        prefix = user + "\n\n"
    return prefix, ptype


# ===========================================================
#  Tasks
# ===========================================================
def run_established(model_name, surveys, fmt, tokenizer, out_dir, limit):
    for sname in surveys:
        cfg = ESTABLISHED[sname]
        survey = json.loads((SURVEY_DIR / f"{sname}.json").read_text())
        items = list(survey.items())
        if limit:
            items = items[:limit]
        result = {
            "model": model_name, "survey": sname, "survey_type": cfg["type"],
            "format": fmt, "scoring": "constrained_logprob_digit",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_items": len(items), "responses": {}, "construct_averages": {},
        }
        for variant, rev in [("normal", False), ("reversed", True)]:
            items_res = {}
            for iid, idata in items:
                content = idata.get("en_neutral", "")
                meta = idata.get("meta_data", {})
                construct = meta.get(cfg["construct_key"], "unknown")
                sem_scoring = meta.get("scoring", "normal")
                prompt = build_likert_prompt(cfg, content, rev, fmt, tokenizer)
                arg, wgt, dprobs, mass = score_likert_item(
                    model_name, prompt, cfg["options"], rev)
                # semantic reverse-keyed items
                def apply_sem(s):
                    if s is None:
                        return None
                    return cfg["scale_max"] + 1 - s if sem_scoring == "reverse" else s
                final_arg = apply_sem(arg)
                final_wgt = apply_sem(wgt)
                items_res[iid] = {
                    "content": content, "construct": construct,
                    "scoring": sem_scoring,
                    "raw_score_argmax": arg,
                    "raw_score_weighted": round(wgt, 4) if wgt is not None else None,
                    "final_score_argmax": final_arg,
                    "final_score": round(final_wgt, 4) if final_wgt is not None else None,
                    "digit_prob_mass": round(mass, 4),
                }
            result["responses"][variant] = items_res
            # construct averages using weighted final_score
            buckets = defaultdict(list)
            for e in items_res.values():
                if e["final_score"] is not None:
                    buckets[e["construct"]].append(e["final_score"])
            result["construct_averages"][variant] = {
                k: round(sum(v) / len(v), 4) for k, v in sorted(buckets.items())}
        # combined
        allc = set(result["construct_averages"]["normal"]) | set(
            result["construct_averages"]["reversed"])
        comb = {}
        for k in sorted(allc):
            vals = [result["construct_averages"][v].get(k)
                    for v in ("normal", "reversed")
                    if result["construct_averages"][v].get(k) is not None]
            comb[k] = round(sum(vals) / len(vals), 4) if vals else None
        result["construct_averages"]["combined"] = comb
        outp = out_dir / "established" / sname / f"{model_slug(model_name)}.json"
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        log(f"  [established/{sname}] saved -> {outp.name}  combined={comb}")


def run_ecological(model_name, fmt, tokenizer, out_dir, limit):
    scenarios = json.loads((OFFICIAL / "surveys" / "VP.json").read_text())
    advice_tpl = (OFFICIAL / "prompts" / "VP_advice.txt").read_text().strip()
    chat_tpl = (OFFICIAL / "prompts" / "VP_chat.txt").read_text().strip()
    outp = out_dir / "ecological" / f"{model_slug(model_name)}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    # resume
    if outp.exists():
        result = json.loads(outp.read_text())
        done = {r["portrait_id"] for r in result.get("results", [])}
    else:
        result = {"model": model_name, "survey": "VP", "format": fmt,
                  "timestamp": datetime.now(timezone.utc).isoformat(),
                  "num_scenarios": len(scenarios), "results": []}
        done = set()
    todo = scenarios if not limit else scenarios[:limit]
    n_new = 0
    for i, sc in enumerate(todo):
        pid = sc["portrait_id"]
        if pid in done:
            continue
        prefix, ptype = format_vp_prefix(
            sc["content"], pid, advice_tpl, chat_tpl, fmt, tokenizer)
        outs = []
        for out in sc["outputs"]:
            lp = echo_response_logprob(model_name, prefix, out["content"])
            outs.append({"output_id": out["id"], "content": out["content"], **lp})
        result["results"].append({
            "portrait_id": pid, "prompt_type": ptype,
            "title": sc["content"].get("title", ""), "outputs": outs})
        done.add(pid)
        n_new += 1
        if n_new % 10 == 0:
            outp.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        norms = [o["normalized_logprob"] for o in outs if o["normalized_logprob"]]
        log(f"  [eco {i+1}/{len(todo)}] pid={pid} ({ptype}) "
            f"norm_lp={[round(x,2) for x in norms]}")
    outp.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log(f"  [ecological] saved -> {outp.name} ({len(result['results'])} scenarios)")


def _disp(name):
    if name in DISPLAY_NAME_MAP:
        return DISPLAY_NAME_MAP[name]
    return name.replace("_", " ")


def run_recognition(model_name, surveys, fmt, tokenizer, out_dir, limit, workers):
    for sname in surveys:
        kind, ckey, sfile = RECOG_SURVEY_CFG[sname]
        defs = RECOG_DEFS[kind]
        outp = out_dir / "recognition" / sname / f"{model_slug(model_name)}.json"
        outp.parent.mkdir(parents=True, exist_ok=True)
        if outp.exists():
            result = json.loads(outp.read_text())
        else:
            result = {"model": model_name, "survey": sname, "format": fmt,
                      "timestamp": datetime.now(timezone.utc).isoformat(),
                      "construct_definitions_used": defs, "results": {}}
        # build items
        if sname == "VP":
            scenarios = json.loads((OFFICIAL / sfile).read_text())
            items = []
            for sc in scenarios:
                pid = sc["portrait_id"]
                for out in sc.get("outputs", []):
                    content = sc.get("content", {}) or {}
                    title = (content.get("title") or "").strip()
                    text = (content.get("text") or "").strip()
                    resp = (out.get("content") or "").strip()
                    qt = (f"Title: {title}\n" if title else "") + \
                        f"Scenario: {text}\nResponse: {resp}"
                    pos = set()
                    for grp_key in ("correlations", "higher_pvq_correlations",
                                    "bfi_correlations"):
                        for nm, val in out.get(grp_key, []):
                            if float(val) >= VP_POSITIVE_THRESHOLD:
                                pos.add(_disp(nm))
                    items.append({"key": f"pid{pid}_out{out['id']}", "qt": qt,
                                  "pid": pid, "oid": out["id"],
                                  "positive": sorted(pos)})
        else:
            survey = json.loads((OFFICIAL / sfile).read_text())
            items = []
            for iid, idata in sorted(survey.items(),
                                     key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0])):
                items.append({"key": f"item_{iid}",
                              "qt": idata.get("en_neutral", "").strip(),
                              "gt": idata.get("meta_data", {}).get(ckey)})
        if limit:
            items = items[:limit]

        cnames = list(defs.keys())

        def process_item(item):
            preds = {}
            for cname in cnames:
                prompt = build_recog_prompt(kind, item["qt"], cname, defs[cname],
                                            fmt, tokenizer)
                parsed, p_yes, p_no = score_yes_no(model_name, prompt)
                preds[cname] = {"parsed": parsed,
                                "p_yes": round(p_yes, 4), "p_no": round(p_no, 4)}
            entry = {"question_text": item["qt"], "predictions": preds}
            if sname == "VP":
                entry.update({"portrait_id": item["pid"], "output_id": item["oid"],
                              "ground_truth_positive": item["positive"]})
            else:
                entry["ground_truth_construct"] = item["gt"]
            return item["key"], entry

        pending = [it for it in items if it["key"] not in result["results"]
                   or any(result["results"][it["key"]]["predictions"].get(c, {}).get("parsed") is None
                          for c in cnames)]
        log(f"  [recognition/{sname}] {len(pending)}/{len(items)} items to do "
            f"({len(cnames)} constructs each)")
        done = 0
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for key, entry in ex.map(process_item, pending):
                    result["results"][key] = entry
                    done += 1
                    if done % 50 == 0:
                        outp.write_text(json.dumps(result, indent=2, ensure_ascii=False))
                        log(f"    ...{done}/{len(pending)}")
        else:
            for it in pending:
                key, entry = process_item(it)
                result["results"][key] = entry
                done += 1
                if done % 50 == 0:
                    outp.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        outp.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        log(f"  [recognition/{sname}] saved -> {outp.name} ({len(result['results'])} items)")


# ===========================================================
#  Main
# ===========================================================
def main():
    global VLLM_PORT, SURVEY_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--format", choices=["chat", "plain"], required=True)
    ap.add_argument("--task", default="all",
                    help="comma of: established,ecological,recognition,all")
    ap.add_argument("--surveys", default="PVQ,BFI44",
                    help="established/recog surveys, comma-separated")
    ap.add_argument("--recog-surveys", default="PVQ,BFI44,VP")
    ap.add_argument("--out-tag", required=True, help="e.g. base / instruct_plain")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="0 = all items")
    ap.add_argument("--recog-workers", type=int, default=8)
    ap.add_argument("--survey-dir", default=None,
                    help="directory with {PVQ,BFI44}.json item files "
                         "(default: cue-reduced surveys)")
    ap.add_argument("--no-server", action="store_true",
                    help="assume vLLM already running on port")
    ap.add_argument("--port", type=int, default=VLLM_PORT,
                    help="vLLM port (MUST be unique per concurrent job on the "
                         "same node; two jobs must not share a port; the second aborts because it finds the first's "
                         "server on the port)")
    args = ap.parse_args()

    VLLM_PORT = args.port
    if args.survey_dir:
        SURVEY_DIR = Path(args.survey_dir)
    log(f"Survey dir: {SURVEY_DIR}")
    log(f"Using vLLM port {VLLM_PORT}")

    (HERE / "logs").mkdir(exist_ok=True)
    out_dir = HERE / "results" / args.out_tag
    tasks = args.task.split(",") if args.task != "all" else \
        ["established", "ecological", "recognition"]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    proc = None
    try:
        if not args.no_server:
            free = get_free_gpus()
            log(f"Free GPUs: {free}")
            proc = launch_vllm(args.model, args.tp, args.max_model_len,
                               args.enforce_eager)
            wait_for_server(proc)

        est_surveys = [s for s in args.surveys.split(",") if s in ESTABLISHED]
        recog_surveys = [s for s in args.recog_surveys.split(",") if s in RECOG_SURVEY_CFG]

        if "established" in tasks:
            log("=== TASK: established ===")
            run_established(args.model, est_surveys, args.format, tokenizer,
                            out_dir, args.limit)
        if "ecological" in tasks:
            log("=== TASK: ecological ===")
            run_ecological(args.model, args.format, tokenizer, out_dir, args.limit)
        if "recognition" in tasks:
            log("=== TASK: recognition ===")
            run_recognition(args.model, recog_surveys, args.format, tokenizer,
                            out_dir, args.limit, args.recog_workers)
    finally:
        if proc is not None:
            log("Shutting down vLLM ...")
            kill_server(proc)
    log("Run complete.")


if __name__ == "__main__":
    main()
