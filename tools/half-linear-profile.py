"""Collect five-process complete-call profiles through the first failed speedup."""

import csv
import os
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    destination = Path(sys.argv[1])
    destination.parent.mkdir(parents=True, exist_ok=True)
    benchmark = Path(__file__).with_name("half-linear-bench.lisp")
    records = []
    thresholds = {}
    full_sweep = os.environ.get("CL_INFERENCE_HALF_BENCH_FULL_SWEEP") == "1"

    def collect(phase, counts):
        environment = dict(os.environ, CL_INFERENCE_HALF_BENCH_COUNTS=",".join(map(str, counts)))
        for process in range(1, 6):
            print(f"{phase}: process {process}/5, counts {counts}", flush=True)
            result = subprocess.run(
                ["sbcl", "--dynamic-space-size", "4096", "--script", str(benchmark)],
                env=environment, text=True, capture_output=True,
            )
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            lines = [line for line in result.stdout.splitlines() if line.startswith("CSV,")]
            if len(lines) != 44 * len(counts):
                raise RuntimeError(f"Incomplete profile: {len(lines)} rows")
            records.extend([phase, process, *line.split(",")[1:]] for line in lines)
            with destination.open("w", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(["phase", "process", "encoding", "weight_rows", "columns",
                                 "activation_rows", "path", "milliseconds", "lisp_bytes"])
                writer.writerows(records)

    collect("core", [1, 6, 46])
    for start in range(1, 65, 8):
        collect(f"sweep-{start}", [n for n in range(start, start + 8) if n not in (1, 6, 46)])
        for encoding in ("f16", "bf16"):
            if encoding in thresholds:
                continue
            for count in range(1, start + 8):
                values = {}
                for _, _, enc, rows, columns, n, path, ms, _ in records:
                    if enc == encoding and int(n) == count:
                        values.setdefault((rows, columns, path), []).append(float(ms))
                shapes = {(rows, columns) for rows, columns, _ in values}
                if any(statistics.median(values[rows, columns, "encoded"])
                       > 0.9 * statistics.median(values[rows, columns, "tiled"])
                       for rows, columns in shapes):
                    thresholds[encoding] = count - 1
                    print(f"{encoding}: threshold {count - 1}, first failure {count}", flush=True)
                    break
        if len(thresholds) == 2 and not full_sweep:
            break
    for encoding in ("f16", "bf16"):
        thresholds.setdefault(encoding, 64)
    print(f"Selected thresholds: {thresholds}", flush=True)


if __name__ == "__main__":
    main()
