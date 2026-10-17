"""GPU check of the residency items, in one pod session.

Run on a Linux pod with two equal NVIDIA cards, from the repository root,
with ``PYTHONPATH=src`` or redact installed::

    python scripts/gpu_check_residency.py all --root /workspace/rescheck
    python scripts/gpu_check_residency.py STEP --root /workspace/rescheck
    python scripts/gpu_check_residency.py report --root /workspace/rescheck

Each step writes ``<root>/results/<step>.json`` (observations and checks)
and its run logs under ``<root>/<step>/``; ``report`` collects them into
``<root>/gpu_check_residency.md``. No hosted model is called.

Pod: 2 x NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB each), a CUDA
13 host (``gpu.minCudaVersion`` 13.0), on ``leonhard_volume`` in EU-RO-1 if
it has stock, else US-PA-1 without it. About $5 an hour; tensor parallel
hangs after NCCL init on this card, which steps 5 and ``second-signal``
need. Optional step 7 needs 4 x RTX PRO 4500 Blackwell, where tensor
parallel works.

Steps (``all`` runs them in this order, about 70 to 80 minutes, $6 to $7):

====================  =========================================================
setup (10-15 min)     Downloads bench-3b, bench-2b, bench-24b (48 GB). Prints
                      versions and ``nvidia-smi``.
order (1 min)         Critique I2: per card, ``torch``'s UUID against
                      ``nvidia-smi``'s, without and with
                      ``CUDA_DEVICE_ORDER=PCI_BUS_ID``. Expected: equal with the
                      variable; record whether they differ without it.
plan (2 min)          Step 2: the plans of [bench-3b, bench-2b] and [bench-24b],
                      detected, against ``gpus=2, gib_per_gpu=<detected>``.
                      Expected: the same lines after the header; header
                      "2 x <name>, 95.6GiB each"; ``gib_per_card`` 2 equal.
copies (20 min)       Step 3: outputs, 1,200 rows, gen=bench-24b, check=bench-2b,
                      default copies, then ``copies: {gen: 1}``. Expected, first
                      run: 24b on cards 0-1 x2 at about 0.68, 2b on card 1 at
                      about 0.22; load events dp 2 and dp 1; three compute
                      apps, two on card 1; rows = ledger = 1,200; exit 0;
                      cards at 0 MiB after. Second run: 24b card 0 at 0.90, 2b
                      card 1 at 0.90. Both runs' tokens per second per model
                      and net stage time are recorded (critique I13); expect
                      the copies run about 1.5 to 1.9 x faster on the stage.
fallback (8 min)      Step 4: a process holds 60 GB on card 1; a script plans
                      bench-3b on cards 0 and 1, applies the plan and loads it,
                      past the run's card check. Expected: the build with 2
                      copies fails (free memory), the WARNING "loading one copy
                      on <uuid of card 0> instead", the failed build's
                      processes ended, one reply, the load event dp 1 with
                      ``fallback`` true. ``/dev/shm`` listed before and after
                      (critique I7): record what a failed build leaves there.
hang (6 min)          Step 5: ``scripts/run.py`` with gen=check=bench-3b-tp2,
                      ``copies: {gen: 1}``; after "vLLM is using nccl" and 60 s
                      of silence, SIGTERM. Expected: "the preload of
                      bench-3b-tp2 has not finished" about 60 s after the
                      signal, then "The load of ... has not returned" and the
                      engine pids; exit code 143 within about 100 s; cards at
                      0 MiB; no VLLM or EngineCore process.
second-signal (4 min) Item 5 C on the same hang: SIGTERM, then SIGTERM again 5 s
                      later. Expected: exit code 143 within about 15 s of the
                      second; the engine processes and TP workers ended; cards
                      at 0 MiB.
healthy (8 min)       Step 6: gen=check=bench-24b, default copies; SIGTERM 20 s
                      after an engine process appears. Expected: exit within
                      about 100 s; either the load ended, or, when it finished
                      within 60 s, a normal release; cards at 0 MiB; nothing
                      left. Then the same run with 200 rows completes (compile
                      cache warnings allowed).
normal-end (3 min)    Step 8: ``generate_outputs`` with bench-3b from a script
                      that returns with the engine loaded. Expected: exit 0
                      within 10 s of the last row; cards at 0 MiB.
kill9 (5 min)         Step 9: ``kill -9`` a run with bench-3b loaded, rerun.
                      Expected: the rerun stops within seconds naming the
                      orphan's pids, before any stage; after killing them by
                      pid, the rerun completes.
tp-copies (15 min)    Step 7, optional, 4 x RTX PRO 4500 only (not in ``all``):
                      gen=check=bench-3b-tp2, no count. Expected: cards 0-3,
                      "x2 copies, each on 2 cards"; load dp 2 tp 2; four
                      workers; 1,200 rows; exit 0.
====================  =========================================================

Not seen here: mixed card sizes (no mixed pod to rent; offline only).
"""

import argparse
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The models of the check, registered in each process that uses them.
BENCH = {
    "bench-3b": {"hf_model_id": "huihui-ai/Llama-3.2-3B-Instruct-abliterated"},
    "bench-2b": {"hf_model_id": "VibeStudio/Nidum-Gemma-2B-Uncensored"},
    "bench-24b": {"hf_model_id": "dphn/Dolphin-Mistral-24B-Venice-Edition"},
    "bench-3b-tp2": {"hf_model_id": "huihui-ai/Llama-3.2-3B-Instruct-abliterated",
                     "min_gpus": 2},
}
#: Gemma's chat template takes no system message.
NO_SYSTEM = {"bench-2b"}
ROWS = 1200
#: EXIT_WAIT_S, two waits of _GRACE_S, vLLM's shutdown, and slack.
EXIT_BOUND_S = 110
#: Seconds the forced exit of a second signal may take.
FORCED_EXIT_S = 20
#: Seconds a script may take to exit after its last row.
NORMAL_END_S = 10
#: Seconds the run's card check may take to stop a run.
PREFLIGHT_S = 60
#: MiB a card may show in use when nothing holds it.
FREE_MIB = 100
#: The exit code of a run ended by SIGTERM.
SIGTERM_EXIT = 128 + signal.SIGTERM
#: How much faster the run with copies must be on the stage.
FASTER = 1.2
STEPS = ["setup", "order", "plan", "copies", "fallback", "hang", "second-signal",
         "healthy", "normal-end", "kill9"]


# -- in the processes the steps start -------------------------------------------

def register_bench() -> None:
    from redact.llms.model_config import VLLMConfig, register_model

    for name, setup in BENCH.items():
        register_model(name, backend_type="vllm", default_max_tokens=256,
                       default_temperature=0.7,
                       supports_system_prompt=name not in NO_SYSTEM,
                       vllm=VLLMConfig(**setup))


def child_run(recipe_path: str, params: str) -> int:
    """``scripts/run.py`` with the bench models registered: its signal
    handling is what the steps check."""
    import importlib.util

    register_bench()
    spec = importlib.util.spec_from_file_location(
        "run_script", ROOT / "scripts" / "run.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script.main([recipe_path, "--params", params])


def child_fallback() -> int:
    """Step 4: plan bench-3b on both cards, apply it and load, past the run's
    card check."""
    import logging

    from redact import residency
    from redact.llms import ModelClient, observe

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    register_bench()
    observe.set_emitter(
        lambda event: print("EVENT", json.dumps(event, default=str), flush=True)
        if event.get("ev") == "local" else None)
    plan = residency.plan_residency(["bench-3b"])
    print(plan.explain(), flush=True)
    residency.apply_plan(plan)
    client = ModelClient.create("bench-3b")
    reply = client.generate([[{"role": "user", "content": "Say hello in one word."}]])
    print("REPLY", json.dumps(reply), flush=True)
    return 0


def child_direct(data_dir: str) -> int:
    """Step 8: generate_outputs from a script that returns with the engine
    loaded; the exit hook releases it."""
    import logging

    from redact import generate_outputs

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    register_bench()
    generate_outputs(data_dir=data_dir, model="bench-3b", check_model="bench-3b",
                     read_back=False)
    print("DONE", time.time(), flush=True)
    return 0


# -- observing the machine ------------------------------------------------------

def smi(query: str) -> list[list[str]]:
    out = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=20).stdout
    return [[part.strip() for part in line.split(",")] for line in out.splitlines()
            if line.strip()]


def cards() -> list[dict]:
    return [{"index": int(i), "uuid": u, "used_mib": int(used), "total_mib": int(total)}
            for i, u, used, total in smi(
                "--query-gpu=index,uuid,memory.used,memory.total")]


def apps() -> list[dict]:
    return [{"pid": int(p), "name": n, "used_mib": m, "uuid": u}
            for p, n, m, u in smi(
                "--query-compute-apps=pid,process_name,used_memory,gpu_uuid")]


def free(listed: list[dict]) -> bool:
    return all(card["used_mib"] < FREE_MIB for card in listed)


def vllm_processes() -> list[dict]:
    """Processes that look like vLLM's engine, coordinator or workers."""
    import psutil

    found = []
    for proc in psutil.process_iter(["pid", "ppid", "name", "cmdline"]):
        text = " ".join([proc.info["name"] or "", *(proc.info["cmdline"] or [])])
        if re.search(r"VLLM|EngineCore|DPCoordinator|DP_Coordinator", text) \
                and proc.info["pid"] != os.getpid():
            found.append({"pid": proc.info["pid"], "ppid": proc.info["ppid"],
                          "name": proc.info["name"], "cmd": text[:120]})
    return found


def shm() -> list[str]:
    with contextlib.suppress(OSError):
        return sorted(os.listdir("/dev/shm"))
    return []


class Sampler:
    """``nvidia-smi`` every 5 s while a run goes on: the most memory per card,
    and the compute apps when the most were listed."""

    def __init__(self):
        self.peak: dict[int, int] = {}
        self.apps: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.wait(5):
            with contextlib.suppress(Exception):
                for card in cards():
                    self.peak[card["index"]] = max(self.peak.get(card["index"], 0),
                                                   card["used_mib"])
                now = apps()
                if len(now) > len(self.apps):
                    self.apps = now

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(10)


def settle(seconds: float = 15) -> list[dict]:
    """The cards once a run is over: wait for them to free, at most this long."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and not free(cards()):
        time.sleep(1)
    return cards()


# -- runs -----------------------------------------------------------------------

def seed(data_dir: Path, rows: int = ROWS) -> None:
    """Harmless input rows in one category of the run's tree."""
    from redact import paths
    from redact.dataset import append_rows, sample_row

    topics = ["a lighthouse", "a garden", "a river", "a library", "a mountain",
              "a bakery", "a train", "a violin", "a comet", "a forest"]
    texts = [f"Write two sentences about {topics[i % len(topics)]} and the number {i}."
             for i in range(rows)]
    path = paths.category_csv("harmless", root=paths.constitution_inputs_dir(data_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    append_rows(path, [sample_row(text, "harmless", turn=0,
                                  extra={"entry_type": "benign"}) for text in texts])


def recipe(step_dir: Path, name: str, models: dict,
           rows: int = ROWS) -> tuple[Path, Path]:
    """A seeded data dir, and the recipe and params of an outputs run in it."""
    data_dir = step_dir / name
    if not (data_dir / "Datasets").exists():
        seed(data_dir, rows)
    path = step_dir / f"{name}.recipe.json"
    params = step_dir / f"{name}.params.json"
    roles = {"translation": None, "paraphraser": None, "paraphrase_check": None,
             "constitution": None}
    path.write_text(json.dumps({"dataset_type": "eval", "data_dir": str(data_dir),
                                "stages": ["outputs"], "models": {**roles, **models},
                                "resume": False}, indent=2))
    params.write_text(json.dumps({"outputs": {"batch_size": 32}}))
    return path, params


class Run:
    """``scripts/run.py`` on a recipe, as its own process, logging to a file."""

    def __init__(self, step_dir: Path, name: str, recipe_path: Path, params: Path):
        self.log = step_dir / f"{name}.log"
        self.data_dir = Path(json.loads(recipe_path.read_text())["data_dir"])
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        self.started = time.monotonic()
        self._fh = self.log.open("w")
        self.proc = subprocess.Popen(
            [sys.executable, __file__, "_run", str(recipe_path), str(params)],
            stdout=self._fh, stderr=subprocess.STDOUT, env=env, cwd=ROOT)

    def text(self) -> str:
        return self.log.read_text(errors="replace")

    def wait_for(self, pattern: str, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if re.search(pattern, self.text()):
                return True
            if self.proc.poll() is not None:
                return bool(re.search(pattern, self.text()))
            time.sleep(1)
        return False

    def wait_silent(self, seconds: float, timeout: float) -> bool:
        """Until the log has not grown for ``seconds``."""
        deadline, size, since = time.monotonic() + timeout, -1, time.monotonic()
        while time.monotonic() < deadline and self.proc.poll() is None:
            now = self.log.stat().st_size
            if now != size:
                size, since = now, time.monotonic()
            elif time.monotonic() - since >= seconds:
                return True
            time.sleep(1)
        return False

    def signal(self, signum=signal.SIGTERM) -> float:
        self.proc.send_signal(signum)
        return time.monotonic()

    def finish(self, timeout: float) -> int | None:
        try:
            code = self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            code = None
        self._fh.close()
        return code

    def trace(self) -> list[dict]:
        traces = sorted((self.data_dir / "Datasets").glob("*.trace.jsonl"))
        if not traces:
            return []
        return [json.loads(line) for line in traces[-1].read_text().splitlines()
                if line.strip()]

    def counts(self) -> dict:
        from redact.dataset import count_rows

        datasets = self.data_dir / "Datasets"
        out = datasets / "output_responses.csv"
        ledger = datasets / "output_responses.state.jsonl"
        lines = ledger.read_text().splitlines() if ledger.exists() else []
        return {"rows": count_rows(out) if out.exists() else 0,
                "acks": sum('"input_id"' in line for line in lines)}


def loads(events: list[dict]) -> list[dict]:
    keys = ("hf_model_id", "data_parallel_size", "tensor_parallel_size",
            "planned_copies", "fallback", "gpu_memory_utilization", "devices",
            "load_ms")
    return [{k: e.get(k) for k in keys} for e in events
            if e.get("ev") == "local" and e.get("phase") == "load"]


def throughput(events: list[dict], stage: str = "outputs") -> dict:
    """Output tokens per second per model over the stage, net of its loads."""
    duration = next((e["duration_s"] for e in events
                     if e.get("ev") == "stage" and e.get("stage") == stage), None)
    load_s = sum(e.get("load_ms", 0) for e in events
                 if e.get("phase") == "load" and e.get("stage") == stage) / 1000
    tokens: dict[str, int] = {}
    for e in events:
        if e.get("ev") == "call" and e.get("stage") == stage and not e.get("error"):
            tokens[e["model"]] = tokens.get(e["model"], 0) + (e.get("out_tok") or 0)
    net = (duration - load_s) if duration else None
    rates = {m: round(t / net) for m, t in tokens.items()} if net else {}
    return {"stage_s": duration, "loads_s": round(load_s, 1),
            "net_s": round(net, 1) if net else None, "out_tokens": tokens,
            "tokens_per_s": rates}


def plan_lines(log: str) -> list[str]:
    return [line for line in log.splitlines()
            if line.startswith(("[residency]", "  group", "    ", "  note", "  PROBLEM",
                                "      ->"))]


# -- the steps ------------------------------------------------------------------

class Result:
    def __init__(self, step: str):
        self.step, self.data, self.checks = step, {}, []

    def check(self, name: str, ok: bool, detail="") -> None:
        self.checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"  [{'ok' if ok else 'FAIL'}] {name} {detail}", flush=True)


def step_setup(r: Result, root: Path) -> None:
    import torch
    import vllm
    from huggingface_hub import snapshot_download

    r.data["versions"] = {"vllm": vllm.__version__, "torch": torch.__version__,
                          "cuda": torch.version.cuda}
    r.data["cards"] = cards()
    r.data["nvidia_smi"] = subprocess.run(["nvidia-smi"], capture_output=True,
                                          text=True).stdout
    for hf_id in dict.fromkeys(setup["hf_model_id"] for setup in BENCH.values()):
        started = time.monotonic()
        snapshot_download(hf_id)
        r.data[f"download_s {hf_id}"] = round(time.monotonic() - started)
    sizes = {card["total_mib"] for card in r.data["cards"]}
    r.check("two equal cards", (len(r.data["cards"]), len(sizes)) == (2, 1),
            r.data["cards"])


def step_order(r: Result, root: Path) -> None:
    code = ("import json, torch; print(json.dumps([str(torch.cuda."
            "get_device_properties(i).uuid) "
            "for i in range(torch.cuda.device_count())]))")
    smi_uuids = [c["uuid"].removeprefix("GPU-").lower() for c in cards()]
    for order in ("unset", "PCI_BUS_ID", "FASTEST_FIRST"):
        env = {k: v for k, v in os.environ.items() if k != "CUDA_DEVICE_ORDER"}
        if order != "unset":
            env["CUDA_DEVICE_ORDER"] = order
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, env=env).stdout.strip().splitlines()[-1]
        torch_uuids = [u.removeprefix("GPU-").lower() for u in json.loads(out)]
        r.data[order] = {"torch": torch_uuids, "nvidia-smi": smi_uuids}
        if order == "PCI_BUS_ID":
            r.check("torch's order with PCI_BUS_ID is nvidia-smi's",
                    torch_uuids == smi_uuids)
        else:
            r.data[f"{order} agrees"] = torch_uuids == smi_uuids


def step_plan(r: Result, root: Path) -> None:
    from redact import residency

    register_bench()
    detected_size = min(c["total_mib"] for c in cards()) / 1024
    for models in (["bench-3b", "bench-2b"], ["bench-24b"]):
        detected = residency.plan_residency(models)
        given = residency.plan_residency(models, gpus=2,
                                         gib_per_gpu=round(detected_size, 1))
        r.data[" + ".join(models)] = {"detected": detected.explain(),
                                      "given": given.explain(),
                                      "gib_per_card": detected.gib_per_card}
        r.check(f"{models}: the same plan detected and given",
                detected.explain().splitlines()[1:] == given.explain().splitlines()[1:])
        header = detected.explain().splitlines()[0]
        r.check(f"{models}: header names 2 cards of one size",
                "detected 2 x " in header and "GiB each" in header, header)


def step_copies(r: Result, root: Path) -> None:
    step_dir = root / "copies"
    step_dir.mkdir(parents=True, exist_ok=True)
    for name, gen in (("shared", "bench-24b"),
                      ("one", {"name": "bench-24b", "copies": 1})):
        path, params = recipe(step_dir, name, {"gen": gen, "check": "bench-2b"})
        with Sampler() as sampled:
            run = Run(step_dir, name, path, params)
            code = run.finish(3600)
        after = settle()
        events = run.trace()
        r.data[name] = {"exit": code, "plan": plan_lines(run.text()),
                        "loads": loads(events), "peak_mib": sampled.peak,
                        "apps_at_most": sampled.apps, "counts": run.counts(),
                        "throughput": throughput(events), "cards_after": after,
                        "left": vllm_processes()}
        d = r.data[name]
        r.check(f"{name}: exit 0", code == 0, code)
        r.check(f"{name}: rows = acks = {ROWS}",
                d["counts"] == {"rows": ROWS, "acks": ROWS}, d["counts"])
        r.check(f"{name}: cards free after", free(after), after)
        r.check(f"{name}: nothing left", not d["left"], d["left"])
        dp = sorted((ld["data_parallel_size"], ld["hf_model_id"]) for ld in d["loads"])
        if name == "shared":
            r.check("shared: 24b dp 2 and 2b dp 1", [n for n, _ in dp] == [1, 2], dp)
            on_card_1 = [a for a in sampled.apps if a["uuid"] == after[1]["uuid"]]
            r.check("shared: three compute apps, two on card 1",
                    (len(sampled.apps), len(on_card_1)) == (3, 2), sampled.apps)
        else:
            r.check("one: both dp 1", [n for n, _ in dp] == [1, 1], dp)
    shared, one = (r.data[n]["throughput"]["net_s"] for n in ("shared", "one"))
    if shared and one:
        r.data["speedup"] = round(one / shared, 2)
        r.check("copies faster on the stage", one / shared > FASTER, r.data["speedup"])


def step_fallback(r: Result, root: Path) -> None:
    step_dir = root / "fallback"
    step_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONUNBUFFERED": "1"}
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import time, torch; x = torch.empty(60 * 2**30, dtype=torch.uint8, "
         "device='cuda:1'); print('holding', flush=True); time.sleep(3600)"],
        stdout=subprocess.PIPE, text=True, env=env)
    try:
        r.check("holder on card 1", holder.stdout.readline().strip() == "holding")
        r.data["cards_held"] = cards()
        r.data["shm_before"] = shm()
        log = step_dir / "fallback.log"
        with log.open("w") as fh:
            code = subprocess.run([sys.executable, __file__, "_fallback"], stdout=fh,
                                  stderr=subprocess.STDOUT, env=env, cwd=ROOT,
                                  timeout=1800).returncode
        text = log.read_text(errors="replace")
        r.data["shm_after"] = shm()
        r.data["shm_new"] = sorted(set(r.data["shm_after"]) - set(r.data["shm_before"]))
        events = [json.loads(line[6:]) for line in text.splitlines()
                  if line.startswith("EVENT ")]
        r.data["loads"] = loads(events)
        r.data["log_tail"] = text.splitlines()[-60:]
        r.data["left"] = vllm_processes()
        r.check("exit 0", code == 0, code)
        r.check("the WARNING names one copy", "loading one copy on" in text)
        r.check("the failed build's processes ended",
                "[vllm] ending engine processes" in text)
        r.check("one reply", "REPLY" in text)
        r.check("load event dp 1, fallback true", any(
            ld["data_parallel_size"] == 1 and ld["fallback"] for ld in r.data["loads"]),
            r.data["loads"])
    finally:
        holder.kill()
        holder.wait(30)
    r.data["cards_after"] = settle()
    r.check("cards free after", free(r.data["cards_after"]), r.data["cards_after"])


def hung_run(root: Path, step: str) -> tuple[Run, bool]:
    step_dir = root / step
    step_dir.mkdir(parents=True, exist_ok=True)
    tp = {"name": "bench-3b-tp2", "copies": 1}
    path, params = recipe(step_dir, "hang", {"gen": tp, "check": "bench-3b-tp2"})
    run = Run(step_dir, "hang", path, params)
    hung = run.wait_for(r"vLLM is using nccl", 900) and run.wait_silent(60, 600)
    return run, hung


def after_exit(r: Result, run: Run, sent: float, code, bound: float) -> None:
    took = round(time.monotonic() - sent, 1)
    markers = ("[residency] the preload", "[vllm] The load", "[vllm] ending",
               "[vllm] engine processes", "Interrupted")
    r.data.update({"exit": code, "exit_s_after_signal": took,
                   "lines": [line for line in run.text().splitlines()
                             if line.startswith(markers)],
                   "cards_after": settle(), "left": vllm_processes()})
    r.check(f"exit code {SIGTERM_EXIT}", code == SIGTERM_EXIT, code)
    r.check(f"exit within {bound} s of the signal", took <= bound, took)
    r.check("cards free after", free(r.data["cards_after"]), r.data["cards_after"])
    r.check("no VLLM or EngineCore process left", not r.data["left"], r.data["left"])


def step_hang(r: Result, root: Path) -> None:
    run, hung = hung_run(root, "hang")
    r.check("the load hangs after NCCL init", hung)
    r.data["during"] = {"cards": cards(), "processes": vllm_processes()}
    sent = run.signal()
    code = run.finish(300)
    after_exit(r, run, sent, code, EXIT_BOUND_S)
    text = run.text()
    r.check("the preload has not finished",
            "the preload of bench-3b-tp2 has not finished" in text)
    r.check("the load has not returned, its processes ended",
            "has not returned" in text and "[vllm] ending engine processes" in text)


def step_second_signal(r: Result, root: Path) -> None:
    run, hung = hung_run(root, "second-signal")
    r.check("the load hangs after NCCL init", hung)
    run.signal()
    time.sleep(5)
    sent = run.signal()
    code = run.finish(120)
    after_exit(r, run, sent, code, FORCED_EXIT_S)
    r.check("the forced exit said so", "Interrupted again" in run.text())


def step_healthy(r: Result, root: Path) -> None:
    import psutil

    step_dir = root / "healthy"
    step_dir.mkdir(parents=True, exist_ok=True)
    path, params = recipe(step_dir, "load", {"gen": "bench-24b", "check": "bench-24b"})
    run = Run(step_dir, "load", path, params)

    def engine_started() -> bool:
        with contextlib.suppress(psutil.Error):
            for child in psutil.Process(run.proc.pid).children(recursive=True):
                with contextlib.suppress(psutil.Error):
                    if "EngineCore" in " ".join(child.cmdline() or [child.name()]):
                        return True
        return False

    deadline = time.monotonic() + 600
    while time.monotonic() < deadline and not engine_started():
        time.sleep(1)
    r.check("an engine process started", engine_started())
    time.sleep(20)
    r.data["during"] = {"cards": cards(), "processes": vllm_processes()}
    sent = run.signal()
    code = run.finish(300)
    after_exit(r, run, sent, code, EXIT_BOUND_S)
    text = run.text()
    r.data["ended_the_load"] = "has not returned" in text
    r.check("the load was ended, or finished and was released",
            r.data["ended_the_load"] or "shutting down" in text)
    models = {"gen": "bench-24b", "check": "bench-24b"}
    path, params = recipe(step_dir, "again", models, rows=200)
    again = Run(step_dir, "again", path, params)
    code = again.finish(3600)
    r.data["again"] = {"exit": code, "counts": again.counts(),
                       "cache_warnings": again.text().count("Failed to reload cubin")}
    r.check("the next run loads and completes", code == 0
            and again.counts() == {"rows": 200, "acks": 200}, r.data["again"])


def step_normal_end(r: Result, root: Path) -> None:
    step_dir = root / "normal-end"
    step_dir.mkdir(parents=True, exist_ok=True)
    data_dir = step_dir / "direct"
    seed(data_dir, 200)
    log = step_dir / "direct.log"
    with log.open("w") as fh:
        proc = subprocess.Popen([sys.executable, __file__, "_direct", str(data_dir)],
                                stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"})
        code = proc.wait(1800)
    ended = time.time()
    done = [float(line.split()[1]) for line in log.read_text(errors="replace")
            .splitlines() if line.startswith("DONE")]
    took = round(ended - done[0], 1) if done else None
    r.data.update({"exit": code, "exit_s_after_last_row": took,
                   "cards_after": settle()})
    r.check("exit 0", code == 0, code)
    r.check(f"exit within {NORMAL_END_S} s of the last row",
            took is not None and took <= NORMAL_END_S, took)
    r.check("cards free after", free(r.data["cards_after"]), r.data["cards_after"])


def step_kill9(r: Result, root: Path) -> None:
    step_dir = root / "kill9"
    step_dir.mkdir(parents=True, exist_ok=True)
    path, params = recipe(step_dir, "first", {"gen": "bench-3b", "check": "bench-3b"})
    run = Run(step_dir, "first", path, params)
    r.check("the engine is loaded", run.wait_for(
        r"\[preload\] bench-3b ready|on the async lane", 900))
    run.signal(signal.SIGKILL)
    run.finish(60)
    time.sleep(10)
    orphans = vllm_processes()
    r.data["orphans"] = orphans
    r.check("kill -9 leaves the engine behind", bool(orphans), orphans)
    rerun = Run(step_dir, "rerun", path, params)
    started = time.monotonic()
    code = rerun.finish(300)
    took = round(time.monotonic() - started, 1)
    text = rerun.text()
    r.data["rerun"] = {"exit": code, "s": took, "message": [
        line for line in text.splitlines()
        if "holds memory" in line or "kill -9" in line]}
    engines = [o for o in orphans if "EngineCore" in o["cmd"]]
    r.check("the rerun stops before any stage, naming the pids",
            code != 0 and all(str(o["pid"]) in text for o in engines), r.data["rerun"])
    r.check(f"within {PREFLIGHT_S} s", took <= PREFLIGHT_S, took)
    for orphan in orphans:
        with contextlib.suppress(ProcessLookupError):
            os.kill(orphan["pid"], signal.SIGKILL)
    settle(30)
    last = Run(step_dir, "last", path, params)
    code = last.finish(3600)
    r.data["last"] = {"exit": code, "counts": last.counts()}
    r.check("after killing them the run completes", code == 0, r.data["last"])


def step_tp_copies(r: Result, root: Path) -> None:
    step_dir = root / "tp-copies"
    step_dir.mkdir(parents=True, exist_ok=True)
    models = {"gen": "bench-3b-tp2", "check": "bench-3b-tp2"}
    path, params = recipe(step_dir, "tp", models)
    with Sampler() as sampled:
        run = Run(step_dir, "tp", path, params)
        code = run.finish(3600)
    events = run.trace()
    r.data.update({"exit": code, "plan": plan_lines(run.text()), "loads": loads(events),
                   "apps_at_most": sampled.apps, "counts": run.counts(),
                   "throughput": throughput(events), "cards_after": settle()})
    r.check("exit 0", code == 0, code)
    r.check("plan: x2 copies, each on 2 cards",
            any("x2 copies (data_parallel_size=2), each on 2 cards" in line
                for line in r.data["plan"]))
    r.check("load dp 2 tp 2", any(
        (ld["data_parallel_size"], ld["tensor_parallel_size"]) == (2, 2)
        for ld in r.data["loads"]), r.data["loads"])
    r.check(f"rows = acks = {ROWS}", r.data["counts"] == {"rows": ROWS, "acks": ROWS},
            r.data["counts"])


def report(root: Path) -> Path:
    lines = ["# GPU check of the residency items", ""]
    for path in sorted((root / "results").glob("*.json")):
        result = json.loads(path.read_text())
        ok = sum(c["ok"] for c in result["checks"])
        lines += [f"## {result['step']}: {ok} of {len(result['checks'])} checks pass",
                  ""]
        lines += [f"- {'ok' if c['ok'] else '**FAIL**'} {c['check']}"
                  + (f": `{json.dumps(c['detail'], default=str)[:300]}`"
                     if c["detail"] not in ("", None) else "")
                  for c in result["checks"]]
        data = json.dumps(result["data"], indent=1, default=str)
        lines += ["", "```json", data[:6000], "```", ""]
    out = root / "gpu_check_residency.md"
    out.write_text("\n".join(lines))
    return out


RUNNERS = {"setup": step_setup, "order": step_order, "plan": step_plan,
           "copies": step_copies, "fallback": step_fallback, "hang": step_hang,
           "second-signal": step_second_signal, "healthy": step_healthy,
           "normal-end": step_normal_end, "kill9": step_kill9,
           "tp-copies": step_tp_copies}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_run":
        return child_run(argv[1], argv[2])
    if argv and argv[0] == "_fallback":
        return child_fallback()
    if argv and argv[0] == "_direct":
        return child_direct(argv[1])
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=[*RUNNERS, "all", "report"])
    parser.add_argument("--root", default="/workspace/rescheck",
                        help="Where the runs, logs and results go.")
    args = parser.parse_args(argv)
    root = Path(args.root)
    (root / "results").mkdir(parents=True, exist_ok=True)
    steps = {"all": STEPS, "report": []}.get(args.step, [args.step])
    for step in steps:
        print(f"== {step}", flush=True)
        result = Result(step)
        started = time.monotonic()
        try:
            RUNNERS[step](result, root)
        except Exception as exc:  # noqa: BLE001 - recorded, and the session goes on
            result.check("ran to the end", False, f"{type(exc).__name__}: {exc}")
        result.data["step_s"] = round(time.monotonic() - started)
        (root / "results" / f"{step}.json").write_text(json.dumps(
            {"step": step, "data": result.data, "checks": result.checks},
            indent=1, default=str))
    print(f"report: {report(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
