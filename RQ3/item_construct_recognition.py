#!/usr/bin/env python3
"""
RQ3: Item-Construct Recognition

Tests whether an LLM can recognize which construct a questionnaire item or
VP scenario-output pair measures, using binary yes/no classification via a
vLLM OpenAI-compatible chat API.

Behavior:
  - Launch one vLLM server per model
  - Resume from existing JSON outputs
  - Save after every completed item for crash recovery
  - Retry parsing up to 3 times when the model does not answer yes/no cleanly

Usage:
    python RQ3/item_construct_recognition.py

Outputs:
    results/RQ3/item_construct_recognition/{survey_name}/{model_slug}.json
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

VLLM_PORT = 8001
GPU_MEMORY_UTILIZATION = 0.85
DTYPE = "bfloat16"

SERVER_LAUNCH_RETRIES = 3
SERVER_CRASH_RESTARTS = 10
CONSECUTIVE_FAIL_LIMIT = 5
MAX_TOKENS = 1024
PARSE_RETRIES = 3
VP_POSITIVE_THRESHOLD = 0.3

SURVEYS = {
    "BFI10": {
        "type": "established",
        "survey_file": ROOT / "surveys" / "BFI10.json",
        "construct_key": "trait",
        "construct_definitions": {
            "Openness": "The tendency to be imaginative, curious, and receptive to new ideas, experiences, and perspectives",
            "Conscientiousness": "The tendency to be organized, responsible, hardworking, and goal-directed",
            "Extraversion": "The tendency to be sociable, assertive, active, and to experience positive emotions",
            "Agreeableness": "The tendency to be compassionate, cooperative, trusting, and helpful toward others",
            "Neuroticism": "The tendency to experience negative emotions such as anxiety, depression, and emotional instability",
        },
    },
    "BFI44": {
        "type": "established",
        "survey_file": ROOT / "surveys" / "BFI44.json",
        "construct_key": "trait",
        "construct_definitions": {
            "Openness": "The tendency to be imaginative, curious, and receptive to new ideas, experiences, and perspectives",
            "Conscientiousness": "The tendency to be organized, responsible, hardworking, and goal-directed",
            "Extraversion": "The tendency to be sociable, assertive, active, and to experience positive emotions",
            "Agreeableness": "The tendency to be compassionate, cooperative, trusting, and helpful toward others",
            "Neuroticism": "The tendency to experience negative emotions such as anxiety, depression, and emotional instability",
        },
    },
    "PVQ": {
        "type": "established",
        "survey_file": ROOT / "surveys" / "PVQ.json",
        "construct_key": "value",
        "construct_definitions": {
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
    },
    "PVQ21": {
        "type": "established",
        "survey_file": ROOT / "surveys" / "PVQ21.json",
        "construct_key": "value",
        "construct_definitions": {
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
    },
    "VP": {
        "type": "vp",
        "survey_file": ROOT / "surveys" / "VP.json",
        "construct_definitions": {
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
            "Openness": "The tendency to be imaginative, curious, and receptive to new ideas, experiences, and perspectives",
            "Conscientiousness": "The tendency to be organized, responsible, hardworking, and goal-directed",
            "Extraversion": "The tendency to be sociable, assertive, active, and to experience positive emotions",
            "Agreeableness": "The tendency to be compassionate, cooperative, trusting, and helpful toward others",
            "Neuroticism": "The tendency to experience negative emotions such as anxiety, depression, and emotional instability",
        },
    },
}

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
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "tensor_parallel": 1,
        "max_model_len": 2048,
    },
    {
        "name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "tensor_parallel": 2,
        "max_model_len": 2048,
        "enforce_eager": True,
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
        "name": "Qwen/Qwen2.5-72B-Instruct",
        "tensor_parallel": 4,
        "max_model_len": 2048,
    },
    {
        "name": "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
        "tensor_parallel": 4,
        "max_model_len": 2048,
        "enforce_eager": True,
    },
]

PROMPT_TEMPLATES = {
    "bfi": ROOT / "prompts" / "item_construct_recognition_bfi.txt",
    "pvq": ROOT / "prompts" / "item_construct_recognition_pvq.txt",
    "vp": ROOT / "prompts" / "item_construct_recognition_vp.txt",
}

SURVEY_PROMPT_MAP = {
    "BFI10": "bfi",
    "BFI44": "bfi",
    "PVQ": "pvq",
    "PVQ21": "pvq",
    "VP": "vp",
}

DISPLAY_NAME_MAP = {
    "Self_Direction": "Self-Direction",
    "Self_Enhancement": "Self-Enhancement",
    "Self_Transcendence": "Self-Transcendence",
    "Openness_to_Change": "Openness to Change",
}

_THINK_RE = None


class ServerCrashError(Exception):
    """Raised when too many consecutive failures indicate server crash."""



def log(msg: str = ""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def model_slug(name: str) -> str:
    return name.split("/")[-1]


def display_construct_name(name: str) -> str:
    if name in DISPLAY_NAME_MAP:
        return DISPLAY_NAME_MAP[name]
    return name.replace("_", " ")


def get_free_gpus(threshold_mb: int = 1000) -> list[int]:
    """Return indices of GPUs using less than threshold_mb MiB."""
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
    gpu_str = ",".join(str(gpu) for gpu in selected_gpus)

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
        "--enable-prefix-caching",
        "--max-num-seqs",
        "8",
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

    return subprocess.Popen(
        cmd,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def wait_for_server(
    proc: subprocess.Popen,
    port: int,
    timeout: int = 3600,
    interval: int = 10,
):
    """Block until the vLLM server responds to /v1/models."""
    url = f"http://localhost:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"vLLM process exited with code {proc.returncode} during startup"
            )
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
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


def launch_with_retry(
    model_cfg: dict,
    max_retries: int = SERVER_LAUNCH_RETRIES,
) -> subprocess.Popen:
    """Launch vLLM server with retries on startup failure."""
    for attempt in range(max_retries):
        proc = None
        try:
            proc = launch_vllm(model_cfg)
            wait_for_server(proc, VLLM_PORT)
            return proc
        except (TimeoutError, Exception) as exc:
            log(f"  Launch attempt {attempt + 1}/{max_retries} failed: {exc}")
            if proc:
                kill_server(proc)
            if attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                log(f"  Waiting {wait}s before retry...")
                time.sleep(wait)

    raise RuntimeError(
        f"Failed to launch {model_cfg['name']} after {max_retries} attempts"
    )


def _extract_reasoning(content: str, message: dict) -> tuple[str, str | None]:
    """Extract reasoning and clean content from model-specific formats."""
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


_active_proc: subprocess.Popen | None = None


def set_active_proc(proc: subprocess.Popen | None):
    global _active_proc
    _active_proc = proc


def _check_server_alive():
    """Raise immediately if the vLLM server process is dead."""
    if _active_proc is not None and _active_proc.poll() is not None:
        raise ServerCrashError(
            f"vLLM process exited (code={_active_proc.returncode})"
        )


def query_model(
    model_name: str,
    messages: list[dict] | str,
    port: int = VLLM_PORT,
    max_retries: int = 5,
    temperature: float = 0.0,
    max_tokens: int = MAX_TOKENS,
) -> dict:
    """Send a chat prompt to vLLM and return parsed response."""
    url = f"http://localhost:{port}/v1/chat/completions"
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    temps_to_try = [temperature]
    if temperature == 0.0:
        temps_to_try.extend([0.3, 0.5, 0.7, 1.0])

    for temp in temps_to_try:
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temp,
        }
        for attempt in range(max_retries):
            _check_server_alive()
            try:
                response = requests.post(url, json=payload, timeout=120)
                response.raise_for_status()
                data = response.json()
                message = data["choices"][0]["message"]
                raw_content = message.get("content")
                if raw_content is None:
                    raise ValueError("Model returned null content")
                content, reasoning = _extract_reasoning(raw_content, message)
                return {"content": content, "reasoning": reasoning}
            except ServerCrashError:
                raise
            except (requests.ConnectionError, ConnectionRefusedError) as exc:
                _check_server_alive()
                if attempt < max_retries - 1:
                    wait = 5 * (attempt + 1)
                    log(f"    Retry {attempt + 1}/{max_retries} temp={temp} (wait {wait}s): {exc}")
                    time.sleep(wait)
                else:
                    if temp != temps_to_try[-1]:
                        log(f"    Falling back to temperature {temps_to_try[temps_to_try.index(temp) + 1]}")
                        break
                    raise
            except Exception as exc:
                if attempt < max_retries - 1:
                    wait = 5 * (attempt + 1)
                    log(f"    Retry {attempt + 1}/{max_retries} temp={temp} (wait {wait}s): {exc}")
                    time.sleep(wait)
                else:
                    if temp != temps_to_try[-1]:
                        log(f"    Falling back to temperature {temps_to_try[temps_to_try.index(temp) + 1]}")
                        break
                    raise


def parse_yes_no(response: str) -> bool | None:
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


def query_and_parse_binary(
    model_name: str,
    prompt_text: str,
    port: int = VLLM_PORT,
) -> tuple[str, bool | None, str | None]:
    """Query model and retry up to PARSE_RETRIES on yes/no parse failure."""
    reasoning = None
    raw_response = ""
    for attempt in range(1 + PARSE_RETRIES):
        resp = query_model(
            model_name=model_name,
            messages=prompt_text,
            port=port,
            temperature=0.0,
            max_tokens=MAX_TOKENS,
        )
        raw_response = resp["content"]
        reasoning = resp["reasoning"]
        parsed = parse_yes_no(raw_response)
        if parsed is not None:
            return raw_response, parsed, reasoning
        if attempt < PARSE_RETRIES:
            log(
                f"          -> parse retry {attempt + 1}/{PARSE_RETRIES}: "
                f"\"{raw_response[:60]}\""
            )
    return raw_response, None, reasoning


def result_path(survey_name: str, model_name: str) -> Path:
    return (
        ROOT
        / "results"
        / "RQ3"
        / "item_construct_recognition"
        / survey_name
        / f"{model_slug(model_name)}.json"
    )


def load_existing_result(survey_name: str, model_name: str) -> dict | None:
    path = result_path(survey_name, model_name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def save_result(survey_name: str, model_name: str, result: dict):
    path = result_path(survey_name, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log(f"  => Saved: {path.relative_to(ROOT)}")


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def build_established_items(survey_name: str) -> list[dict]:
    cfg = SURVEYS[survey_name]
    survey_data = load_json(cfg["survey_file"])
    items = []
    for item_id, item_data in sorted(
        survey_data.items(),
        key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0]),
    ):
        question_text = item_data.get("en_neutral", "").strip()
        meta = item_data.get("meta_data", {})
        ground_truth_construct = meta.get(cfg["construct_key"])
        items.append(
            {
                "item_key": f"item_{item_id}",
                "item_id": item_id,
                "question_text": question_text,
                "ground_truth_construct": ground_truth_construct,
                "construct_definitions": cfg["construct_definitions"],
            }
        )
    return items


def build_vp_question_text(scenario: dict, output: dict) -> str:
    lines = []
    content = scenario.get("content", {}) or {}
    title = (content.get("title") or "").strip()
    text = (content.get("text") or "").strip()
    response_text = (output.get("content") or "").strip()

    if title:
        lines.append(f"Title: {title}")
    lines.append(f"Scenario: {text}")
    lines.append(f"Response: {response_text}")
    return "\n".join(lines)


def normalize_correlation_pairs(pairs: list[list]) -> dict[str, float]:
    return {
        display_construct_name(name): float(value)
        for name, value in pairs
    }


def build_vp_ground_truth(output: dict) -> tuple[dict[str, dict[str, float]], list[str]]:
    raw = {
        "pvq_values": normalize_correlation_pairs(output.get("correlations", [])),
        "higher_order_values": normalize_correlation_pairs(
            output.get("higher_pvq_correlations", [])
        ),
        "bfi_traits": normalize_correlation_pairs(output.get("bfi_correlations", [])),
    }

    positive = set()
    for group_values in raw.values():
        for construct_name, corr_value in group_values.items():
            if corr_value >= VP_POSITIVE_THRESHOLD:
                positive.add(construct_name)
    return raw, sorted(positive)


def build_vp_items() -> list[dict]:
    scenarios = load_json(SURVEYS["VP"]["survey_file"])
    items = []
    for scenario in scenarios:
        portrait_id = scenario["portrait_id"]
        for output in scenario.get("outputs", []):
            raw_ground_truth, positive_constructs = build_vp_ground_truth(output)
            items.append(
                {
                    "item_key": f"pid{portrait_id}_out{output['id']}",
                    "portrait_id": portrait_id,
                    "output_id": output["id"],
                    "question_text": build_vp_question_text(scenario, output),
                    "ground_truth_constructs": raw_ground_truth,
                    "ground_truth_positive": positive_constructs,
                    "construct_definitions": SURVEYS["VP"]["construct_definitions"],
                }
            )
    return items


_PROMPT_CACHE: dict[str, str] = {}


def _load_prompt_template(template_key: str) -> str:
    if template_key not in _PROMPT_CACHE:
        path = PROMPT_TEMPLATES[template_key]
        _PROMPT_CACHE[template_key] = path.read_text().strip()
    return _PROMPT_CACHE[template_key]


def build_prompt(
    survey_name: str,
    question_text: str,
    construct_name: str,
    construct_definition: str,
) -> str:
    template_key = SURVEY_PROMPT_MAP[survey_name]
    template = _load_prompt_template(template_key)
    return template.format(
        question_text=question_text,
        construct_name=construct_name,
        construct_definition=construct_definition,
    )


def init_result(model_name: str, survey_name: str, construct_definitions: dict) -> dict:
    return {
        "model": model_name,
        "survey": survey_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "construct_definitions_used": dict(construct_definitions),
        "results": {},
    }


def ensure_result_shell(existing: dict | None, model_name: str, survey_name: str) -> dict:
    construct_definitions = SURVEYS[survey_name]["construct_definitions"]
    if existing is None:
        return init_result(model_name, survey_name, construct_definitions)

    existing.setdefault("model", model_name)
    existing.setdefault("survey", survey_name)
    existing.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    existing.setdefault("construct_definitions_used", dict(construct_definitions))
    existing.setdefault("results", {})
    return existing


def prediction_is_complete(prediction: dict | None) -> bool:
    if not prediction:
        return False
    if prediction.get("raw_response") in ("", None):
        return False
    if prediction.get("parsed") is None:
        return False
    return True


def item_needs_repair(item_result: dict | None, construct_names: list[str]) -> bool:
    if not item_result:
        return True
    predictions = item_result.get("predictions", {})
    for construct_name in construct_names:
        if not prediction_is_complete(predictions.get(construct_name)):
            return True
    return False


def count_complete_items(items: list[dict], results: dict) -> int:
    complete = 0
    for item in items:
        construct_names = list(item["construct_definitions"].keys())
        if not item_needs_repair(results.get(item["item_key"]), construct_names):
            complete += 1
    return complete


def expected_item_count(survey_name: str) -> int:
    if survey_name == "VP":
        survey_data = load_json(SURVEYS["VP"]["survey_file"])
        return sum(len(scenario.get("outputs", [])) for scenario in survey_data)
    survey_data = load_json(SURVEYS[survey_name]["survey_file"])
    return len(survey_data)


def pending_surveys_for_model(model_name: str) -> list[str]:
    pending = []
    for survey_name in SURVEYS:
        existing = load_existing_result(survey_name, model_name)
        results = existing.get("results", {}) if existing else {}
        items = build_vp_items() if survey_name == "VP" else build_established_items(survey_name)
        done = count_complete_items(items, results)
        if done < len(items):
            pending.append(survey_name)
    return pending


def run_established_survey(
    model_name: str,
    survey_name: str,
    port: int = VLLM_PORT,
):
    items = build_established_items(survey_name)
    existing = load_existing_result(survey_name, model_name)
    result = ensure_result_shell(existing, model_name, survey_name)
    completed_items = count_complete_items(items, result["results"])

    log(f"  Resuming {survey_name}: {completed_items}/{len(items)} items complete")
    consecutive_fails = 0

    for idx, item in enumerate(items, start=1):
        item_key = item["item_key"]
        existing_item = result["results"].get(item_key)
        existing_predictions = existing_item.get("predictions", {}) if existing_item else {}
        construct_names = list(item["construct_definitions"].keys())
        if not item_needs_repair(existing_item, construct_names):
            continue

        predictions = dict(existing_predictions)
        for construct_name, construct_definition in item["construct_definitions"].items():
            if prediction_is_complete(existing_predictions.get(construct_name)):
                continue
            prompt_text = build_prompt(
                survey_name=survey_name,
                question_text=item["question_text"],
                construct_name=construct_name,
                construct_definition=construct_definition,
            )
            try:
                raw_response, parsed, reasoning = query_and_parse_binary(
                    model_name=model_name,
                    prompt_text=prompt_text,
                    port=port,
                )
                consecutive_fails = 0
            except ServerCrashError:
                save_result(survey_name, model_name, result)
                raise
            except Exception as exc:
                consecutive_fails += 1
                log(
                    f"    [{survey_name}] {item_key} / {construct_name} "
                    f"[ERR] query failed: {exc}"
                )
                if consecutive_fails >= CONSECUTIVE_FAIL_LIMIT:
                    save_result(survey_name, model_name, result)
                    raise ServerCrashError(
                        f"Server unresponsive ({consecutive_fails} consecutive failures)"
                    )
                raw_response = ""
                parsed = None
                reasoning = None

            pred_entry = {
                "raw_response": raw_response,
                "parsed": parsed,
                "reasoning": reasoning,
            }
            predictions[construct_name] = pred_entry

        result["results"][item_key] = {
            "question_text": item["question_text"],
            "ground_truth_construct": item["ground_truth_construct"],
            "predictions": predictions,
        }
        save_result(survey_name, model_name, result)
        log(
            f"    [{survey_name}] {idx}/{len(items)} {item_key} "
            f"saved ({len(predictions)} constructs)"
        )


def run_vp_survey(
    model_name: str,
    port: int = VLLM_PORT,
):
    items = build_vp_items()
    survey_name = "VP"
    existing = load_existing_result(survey_name, model_name)
    result = ensure_result_shell(existing, model_name, survey_name)
    completed_items = count_complete_items(items, result["results"])

    log(f"  Resuming VP: {completed_items}/{len(items)} items complete")
    consecutive_fails = 0

    for idx, item in enumerate(items, start=1):
        item_key = item["item_key"]
        existing_item = result["results"].get(item_key)
        existing_predictions = existing_item.get("predictions", {}) if existing_item else {}
        construct_names = list(item["construct_definitions"].keys())
        if not item_needs_repair(existing_item, construct_names):
            continue

        predictions = dict(existing_predictions)
        for construct_name, construct_definition in item["construct_definitions"].items():
            if prediction_is_complete(existing_predictions.get(construct_name)):
                continue
            prompt_text = build_prompt(
                survey_name=survey_name,
                question_text=item["question_text"],
                construct_name=construct_name,
                construct_definition=construct_definition,
            )
            try:
                raw_response, parsed, reasoning = query_and_parse_binary(
                    model_name=model_name,
                    prompt_text=prompt_text,
                    port=port,
                )
                consecutive_fails = 0
            except ServerCrashError:
                save_result(survey_name, model_name, result)
                raise
            except Exception as exc:
                consecutive_fails += 1
                log(
                    f"    [VP] {item_key} / {construct_name} "
                    f"[ERR] query failed: {exc}"
                )
                if consecutive_fails >= CONSECUTIVE_FAIL_LIMIT:
                    save_result(survey_name, model_name, result)
                    raise ServerCrashError(
                        f"Server unresponsive ({consecutive_fails} consecutive failures)"
                    )
                raw_response = ""
                parsed = None
                reasoning = None

            predictions[construct_name] = {
                "raw_response": raw_response,
                "parsed": parsed,
                "reasoning": reasoning,
            }

        result["results"][item_key] = {
            "portrait_id": item["portrait_id"],
            "output_id": item["output_id"],
            "question_text": item["question_text"],
            "ground_truth_constructs": item["ground_truth_constructs"],
            "ground_truth_positive": item["ground_truth_positive"],
            "predictions": predictions,
        }
        save_result(survey_name, model_name, result)
        log(
            f"    [VP] {idx}/{len(items)} {item_key} "
            f"saved ({len(predictions)} constructs)"
        )


def _server_is_healthy(port: int = VLLM_PORT) -> bool:
    """Quick health-check: can we reach /v1/models?"""
    try:
        r = requests.get(f"http://localhost:{port}/v1/models", timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def _ensure_server(
    model_cfg: dict,
    proc: subprocess.Popen | None,
) -> subprocess.Popen:
    """Return a running vLLM server, reusing *proc* if still healthy."""
    if proc is not None and proc.poll() is None and _server_is_healthy():
        return proc
    if proc is not None:
        kill_server(proc)
        set_active_proc(None)
        time.sleep(5)
    new_proc = launch_with_retry(model_cfg)
    set_active_proc(new_proc)
    return new_proc


def _run_all_surveys_for_model(
    model_cfg: dict,
    pending_surveys: list[str],
    max_crash_restarts: int = SERVER_CRASH_RESTARTS,
) -> bool:
    """Launch vLLM ONCE and run every pending survey before shutting down.

    The server is only restarted on crash or preventive-restart signals.
    """
    model_name = model_cfg["name"]
    proc: subprocess.Popen | None = None
    crash_count = 0

    try:
        proc = _ensure_server(model_cfg, proc)

        survey_idx = 0
        while survey_idx < len(pending_surveys):
            survey_name = pending_surveys[survey_idx]

            existing = load_existing_result(survey_name, model_name)
            results = existing.get("results", {}) if existing else {}
            items = (
                build_vp_items()
                if survey_name == "VP"
                else build_established_items(survey_name)
            )
            done = count_complete_items(items, results)
            if done >= len(items):
                log(f"  Survey {survey_name} already complete, skipping.")
                survey_idx += 1
                continue

            log(f"{'·' * 40}")
            log(f"  Running survey: {survey_name}")
            log(f"{'·' * 40}")

            try:
                if survey_name == "VP":
                    run_vp_survey(model_name)
                else:
                    run_established_survey(model_name, survey_name)
                survey_idx += 1

            except ServerCrashError as exc:
                crash_count += 1
                log(f"  SERVER CRASH during {survey_name}: {exc}")
                if proc:
                    kill_server(proc)
                    proc = None
                set_active_proc(None)
                if crash_count >= max_crash_restarts:
                    log(
                        f"  Too many crashes ({crash_count}). "
                        f"Giving up on remaining surveys for {model_name}."
                    )
                    return False
                wait = min(30 * crash_count, 120)
                log(f"  Auto-restart (crash #{crash_count}) in {wait}s ...")
                time.sleep(wait)
                proc = _ensure_server(model_cfg, proc)

    except Exception:
        raise
    finally:
        if proc:
            kill_server(proc)
            set_active_proc(None)

    return True


def main():
    log("=" * 60)
    log("  RQ3: Item-Construct Recognition")
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

        pending_surveys = pending_surveys_for_model(model_name)
        if not pending_surveys:
            log(f"SKIP {model_name}: all surveys already complete.")
            continue

        log(f"{'─' * 50}")
        log(f"  Model: {model_name}")
        log(f"  Pending surveys: {pending_surveys}")
        log(f"{'─' * 50}")

        try:
            _run_all_surveys_for_model(model_cfg, pending_surveys)
        except Exception as exc:
            log(f"ERROR with {model_name}: {exc}")
            import traceback

            traceback.print_exc()

    log("=" * 60)
    log("  All done!")
    log("=" * 60)


if __name__ == "__main__":
    main()
