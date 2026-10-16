"""Record three-process half-dispatch profiles for both native threading settings."""

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time


def git(*args, directory):
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


def check_cpu_load(excluded=()):
    ignored = {*excluded, os.getpid()}

    def snapshot():
        output = subprocess.check_output(
            ["ps", "-axo", "pid=,stat=,pcpu=,comm="], text=True)
        busy = {}
        for line in output.splitlines():
            pid, state, cpu, command = line.split(None, 3)
            if int(pid) not in ignored and "T" not in state and float(cpu) >= 50:
                busy[int(pid)] = Path(command).name
        return busy

    busy = snapshot()
    if busy:
        for _ in range(2):
            time.sleep(1)
            current = snapshot()
            busy = {pid: name for pid, name in busy.items() if pid in current}
        if busy:
            raise RuntimeError(f"Unrelated CPU-bound processes prevent isolated profiling: {busy}")


def main():
    destination = Path(sys.argv[1])
    destination.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent.parent
    simd = root.parent / "trivial-simd"
    library = simd / "build/libtrivial_simd.dylib"
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "macos": subprocess.check_output(["sw_vers"], text=True).strip(),
        "revisions": {name: git("rev-parse", "HEAD", directory=root.parent / name)
                      for name in ("cl-inference", "cl-tensor", "trivial-simd")},
        "inference_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "-C", str(root), "diff"])).hexdigest(),
        "native_library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("local/blocks.lisp", "tests/local/blocks.lisp", "tests/local/execution.lisp",
                         "tools/parallel-bench.lisp", "tools/parallel-profile.py")
        },
        "profiles": [],
        "completed_processes": [],
        "cpu_check": "Reject unrelated processes using >=50% CPU in three consecutive snapshots; "
                     "check before/after each process and at result boundaries after 60 seconds.",
    }
    with destination.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["native_threads", "process", "kind", "name", "count", "workers",
                         "path", "milliseconds", "lisp_bytes", "tokens_per_second"])
        for threading in ("single", "default"):
            for process in range(1, 4):
                check_cpu_load()
                environment = dict(os.environ, CL_INFERENCE_PARALLEL_HALF_PROFILE="1")
                for key in ("CL_INFERENCE_PARALLEL_MODELS", "CL_INFERENCE_PARALLEL_PATHS"):
                    environment.pop(key, None)
                if threading == "single":
                    environment["VECLIB_MAXIMUM_THREADS"] = "1"
                else:
                    environment.pop("VECLIB_MAXIMUM_THREADS", None)
                log = Path(tempfile.gettempdir()) / f"{destination.stem}.{threading}-{process}.log"
                print(f"{threading}: process {process}/3; {log}", flush=True)
                rows = 0
                with log.open("w") as output:
                    child = subprocess.Popen(
                        ["sbcl", "--dynamic-space-size", "3GB", "--script",
                         str(root / "tools/parallel-bench.lisp")],
                        cwd=root, env=environment, text=True, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    )
                    checked = time.monotonic()
                    try:
                        for line in child.stdout:
                            output.write(line)
                            output.flush()
                            if time.monotonic() - checked >= 60:
                                check_cpu_load((child.pid,))
                                checked = time.monotonic()
                            if line.startswith("RESULT,"):
                                writer.writerow([threading, process, *line.strip().split(",")[1:]])
                                stream.flush()
                                rows += 1
                            elif line.startswith("PROFILE "):
                                metadata["profiles"].append(line.strip())
                        result = child.wait()
                    except BaseException:
                        child.terminate()
                        child.wait()
                        raise
                check_cpu_load()
                if result or rows != 936:
                    raise RuntimeError(f"Incomplete profile: exit={result}, rows={rows}; see {log}")
                metadata["completed_processes"].append({"native_threads": threading,
                                                        "process": process, "rows": rows})
                metadata["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
                print(f"{threading}: process {process}/3 complete, {rows} rows", flush=True)


if __name__ == "__main__":
    main()
