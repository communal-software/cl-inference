"""Report native llama.cpp activation-quantization drift from F32-dequantized execution."""
import json
import math
import pathlib

root = pathlib.Path(__file__).resolve().parents[2]
manifest = json.loads((root / "tests/acceptance/quant-models.json").read_text())
oracle = root / "reference/oracle"


def trace(path):
    rows = {}
    for line in (path / "trace.txt").read_text().splitlines():
        name, count, *values = line.split()
        if len(values) != int(count):
            raise RuntimeError(f"Malformed trace: {path}")
        rows[name] = list(map(float, values))
    return rows


print("| Model | Format | Worst layer relative error | Worst logit relative error | Greedy IDs agree in every case |")
print("|---|---|---:|---:|---|")
for key, entry in sorted(manifest.items()):
    family, dtype = key.split("-", 1)
    layers, logits, greedy = [], [], True
    for mode in (["plain", "chat", "thinking"] if family == "qwen" else ["plain", "chat"]):
        case = key + "-" + mode
        paths = (oracle / case, oracle / (case + "-native"))
        for index, path in enumerate(paths):
            provenance = json.loads((path / "provenance.json").read_text())
            if any(provenance[field] != entry[field] for field in ("sha256", "source_sha256", "llama_cpp", "dtype")) or provenance["native"] != bool(index):
                raise RuntimeError(f"Oracle provenance differs: {path}")
        reference, native = map(trace, paths)
        if reference.keys() != native.keys():
            raise RuntimeError(f"Trace fields differ: {case}")
        for name, expected in reference.items():
            actual = native[name]
            if len(actual) != len(expected):
                raise RuntimeError(f"Trace dimensions differ: {case}")
            relative = math.sqrt(sum((a-b)**2 for a, b in zip(actual, expected)) / sum(b*b for b in expected))
            (logits if name == "result_output" else layers).append(relative)
        greedy &= (paths[0] / "greedy.ids").read_text() == (paths[1] / "greedy.ids").read_text()
    print(f"| {family} | {dtype.upper()} | {max(layers):.6f} | {max(logits):.6f} | {'yes' if greedy else 'no'} |")
