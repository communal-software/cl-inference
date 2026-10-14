"""Build pinned Q4/Q6 models and F32/native llama.cpp oracle dumps."""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess

root = pathlib.Path(__file__).resolve().parents[2]
reference = root / "reference"
source = reference / "llama.cpp"
commit = "8bb3cf0d216a5ae48360179f2931daf284be4d01"
models = {
    "llama": ("Llama-3.2-1B-Instruct", "1f33ad43d2b85b908ff06fe7002b69806a57359b9b2617ca27d7bdea428ae146"),
    "qwen": ("Qwen3-0.6B", "25fb880334987a71db728fb43af8568a4682208059f754a2735781f5d0970143"),
}
formats = {"q4_0": "Q4_0", "q4_k": "Q4_K_M", "q6_k": "Q6_K"}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--family", choices=models, action="append")
parser.add_argument("--format", choices=formats, action="append")
parser.add_argument("--native", action="store_true", help="Also dump native packed execution for drift comparisons")
args = parser.parse_args()


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(command, log, **kwargs):
    print(" ".join(map(str, command)), flush=True)
    with log.open("w") as stream:
        subprocess.run(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT, check=True, **kwargs)


reference.mkdir(exist_ok=True)
work = reference / "quant-build"
work.mkdir(exist_ok=True)
if not source.exists():
    run(["git", "clone", "--depth", "1", "--branch", "b11574", "https://github.com/ggml-org/llama.cpp.git", source], work / "clone.log")
if subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() != commit:
    raise RuntimeError("llama.cpp does not match the pinned commit")
destination = source / "examples/dump-trace"
destination.mkdir(exist_ok=True)
for path in (root / "tests/acceptance/oracle").iterdir():
    shutil.copy2(path, destination / path.name)
cmake = source / "examples/CMakeLists.txt"
if "add_subdirectory(dump-trace)" not in cmake.read_text():
    with cmake.open("a") as stream:
        stream.write("\nadd_subdirectory(dump-trace)\n")
run(["cmake", "-S", source, "-B", source / "build", "-DGGML_METAL=OFF", "-DGGML_BLAS=OFF",
     "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_SERVER=OFF"], work / "configure.log")
run(["cmake", "--build", source / "build", "--target", "llama-quantize", "llama-dump-trace", "-j8"], work / "build.log")
manifest_path = root / "tests/acceptance/quant-models.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
scratch = work / "f32-scratch.gguf"
for family in args.family or models:
    stem, input_sha = models[family]
    original = reference / "models" / (stem + "-f16.gguf")
    if not original.exists() or sha(original) != input_sha:
        raise RuntimeError(f"Missing or unverified F16 input: {original}")
    for kind in args.format or formats:
        model = original.with_name(stem + "-" + formats[kind] + ".gguf")
        key = family + "-" + kind
        if not model.exists():
            run([source / "build/bin/llama-quantize", original, model, formats[kind], "4"], work / (key + "-quantize.log"))
        digest = sha(model)
        entry = {"file": model.name, "sha256": digest, "source_sha256": input_sha, "llama_cpp": commit, "dtype": kind}
        if key in manifest and manifest[key] != entry:
            raise RuntimeError(f"Generated model does not match its pin: {model}")
        manifest[key] = entry
        run([source / "build/bin/llama-quantize", "--allow-requantize", model, scratch, "F32", "4"], work / (key + "-dequantize.log"))
        modes = ["plain", "chat"] + (["thinking"] if family == "qwen" else [])
        for mode in modes:
            for native in ([False, True] if args.native else [False]):
                case = key + "-" + mode + ("-native" if native else "")
                output = reference / "oracle" / case
                output.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, DUMP_DIR=str(output), DUMP_N="8")
                for variable in ("DUMP_USER", "DUMP_SYSTEM", "DUMP_THINKING"):
                    env.pop(variable, None)
                if mode != "plain":
                    env.update(DUMP_USER="What is the capital of France?", DUMP_SYSTEM="Answer in one word.", DUMP_THINKING="1" if mode == "thinking" else "0")
                run([source / "build/bin/llama-dump-trace", "-m", model if native else scratch,
                     "-p", "The capital of France is" if mode == "plain" else "", "-t", "4", "-ngl", "0",
                     "-c", "512", "--seed", "0"], output / "log.txt", env=env)
                (output / "provenance.json").write_text(json.dumps(dict(entry, native=native), indent=2) + "\n")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
