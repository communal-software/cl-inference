"""Compare pinned SIMD revisions with five alternating CPU inference process pairs."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import runpy
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--before", default="cd61d43c2986c05ac32503d7619b2e7b67d1b584")
    parser.add_argument("--after", default="26bb84d9afaeb146b5e92c1b27f594268914e159")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    support = runpy.run_path(str(root / "tools/parallel-profile.py"))
    git, check_cpu_load = support["git"], support["check_cpu_load"]
    source = root.parent / "trivial-simd"
    work = Path(tempfile.mkdtemp(prefix="simd-inference-"))
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "runtime": subprocess.check_output(["sbcl", "--version"], text=True).strip(),
        "revisions": {name: git("rev-parse", "HEAD", directory=root.parent / name)
                      for name in ("cl-inference", "cl-tensor")},
        "source_sha256": {name: digest(root / name) for name in
                          ("tools/parallel-bench.lisp", "tools/simd-inference-profile.py",
                           "tools/parallel-profile.py")},
        "variants": {}, "processes": [], "work_directory": str(work),
        "method": "Five alternating fresh-process pairs; serial executor; VECLIB_MAXIMUM_THREADS=1; "
                  "automatic dispatch; one warmup and median of three calls with full GC before each. "
                  "Decode overwrites a prepared prefix; generation resets the session. "
                  "Pinned model SHA256 and four greedy token IDs checked for every prompt/variant. "
                  "Independent ASDF caches; matching Release builds; CPU load guard before/after "
                  "each process and at output boundaries after 60 seconds.",
    }
    for variant, revision in (("before", args.before), ("after", args.after)):
        revision = git("rev-parse", revision, directory=source)
        directory = work / variant
        directory.mkdir()
        archive = subprocess.check_output(["git", "-C", str(source), "archive", revision])
        subprocess.run(["tar", "-x", "-C", str(directory)], input=archive, check=True)
        with (work / f"{variant}-build.log").open("w") as log:
            subprocess.run(["cmake", "-S", str(directory), "-B", str(directory / "build"),
                            "-DCMAKE_BUILD_TYPE=Release"], stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run(["cmake", "--build", str(directory / "build"), "--target", "trivial_simd", "-j", "4"],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        metadata["variants"][variant] = {"revision": revision, "source": str(directory),
            "native_library_sha256": digest(directory / "build/libtrivial_simd.dylib"),
            "cmake_cache": (directory / "build/CMakeCache.txt").read_text()}
    expected_verification = None
    results = []
    with args.destination.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["pair", "variant", "kind", "model", "prompt_tokens", "workers",
                         "path", "milliseconds", "lisp_bytes", "tokens_per_second"])
        for pair in range(1, 6):
            for variant in (("before", "after") if pair % 2 else ("after", "before")):
                check_cpu_load()
                environment = dict(os.environ)
                for key in list(environment):
                    if key.startswith("CL_INFERENCE_PARALLEL_"):
                        environment.pop(key)
                environment.update(
                    VECLIB_MAXIMUM_THREADS="1", TRIVIAL_SIMD_BACKEND="native",
                    CL_INFERENCE_SIMD_SOURCE=str(work / variant) + "/",
                    CL_INFERENCE_PARALLEL_MODELS_ONLY="1", CL_INFERENCE_PARALLEL_SERIAL_ONLY="1",
                    CL_INFERENCE_PARALLEL_PATHS="auto", CL_INFERENCE_PARALLEL_VERIFY="1",
                    ASDF_OUTPUT_TRANSLATIONS=f'(:output-translations (t ("{work / (variant + "-cache")}/" :implementation)) :ignore-inherited-configuration)')
                log_path = work / f"pair-{pair}-{variant}.log"
                print(f"Pair {pair}/5 {variant}: {log_path}", flush=True)
                rows, verification, profiles = [], [], []
                with log_path.open("w") as log:
                    child = subprocess.Popen(["sbcl", "--dynamic-space-size", "4096", "--script",
                                              str(root / "tools/parallel-bench.lisp")], cwd=root,
                                             env=environment, stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT, text=True)
                    checked = time.monotonic()
                    try:
                        for line in child.stdout:
                            log.write(line)
                            log.flush()
                            if time.monotonic() - checked >= 60:
                                check_cpu_load((child.pid,))
                                checked = time.monotonic()
                            if line.startswith("RESULT,"):
                                row = line.strip().split(",")[1:]
                                if len(row) != 8 or float(row[5]) <= 0 or row[3:5] != ["0", "AUTO"]:
                                    raise RuntimeError(f"Invalid benchmark row: {row}")
                                rows.append(row)
                            elif line.startswith("VERIFY,"):
                                verification.append(line.strip())
                            elif line.startswith("PROFILE "):
                                profiles.append(line.strip())
                        code = child.wait()
                    except BaseException:
                        child.terminate()
                        child.wait()
                        raise
                check_cpu_load()
                keys = {(r[0], r[1], r[2]) for r in rows}
                required = {(kind, model, str(count)) for kind in ("prefill", "decode", "generate")
                            for model in ("F16", "Q8_0", "QWEN-F16", "QWEN-Q8") for count in (16, 128)}
                if code or len(rows) != 24 or keys != required or len(verification) != 8:
                    raise RuntimeError(f"Incomplete profile: exit={code}, rows={len(rows)}; see {log_path}")
                if len(profiles) != 1 or 'simd=:NATIVE' not in profiles[0] or str(work / variant) not in profiles[0]:
                    raise RuntimeError(f"Unexpected backend/source: {profiles}")
                if expected_verification is None:
                    expected_verification = verification
                if verification != expected_verification:
                    raise RuntimeError(f"Greedy token mismatch: {log_path}")
                writer.writerows([pair, variant, *row] for row in rows)
                stream.flush()
                results.extend((pair, variant, row) for row in rows)
                metadata["processes"].append({"pair": pair, "variant": variant,
                                               "profile": profiles[0], "verification": verification,
                                               "log_sha256": digest(log_path)})
                metadata["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                args.destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
                print(f"Pair {pair}/5 {variant}: 24 cases verified", flush=True)
    summary = []
    for key in sorted(required):
        values = {variant: [float(row[5]) for _, v, row in results
                            if v == variant and tuple(row[:3]) == key] for variant in ("before", "after")}
        gains = [100 * (1 - after / before) for before, after in zip(values["before"], values["after"])]
        summary.append({"kind": key[0], "model": key[1], "prompt_tokens": int(key[2]),
                        "before_ms": statistics.median(values["before"]),
                        "after_ms": statistics.median(values["after"]),
                        "paired_gain_percent": statistics.median(gains),
                        "pair_min_percent": min(gains), "pair_max_percent": max(gains),
                        "pairs_at_least_5_percent_faster": sum(g >= 5 for g in gains),
                        "pairs_at_least_5_percent_slower": sum(g <= -5 for g in gains)})
    args.destination.with_name(args.destination.stem + "-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Complete: {args.destination}; logs: {work}", flush=True)


if __name__ == "__main__":
    main()
