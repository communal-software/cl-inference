"""Prepare pinned Gemma Q8_0/F32-dequantized llama.cpp acceptance dumps."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT / "tests/acceptance/gemma-model.json").read_text())
REFERENCE = ROOT / "reference"
SOURCE = REFERENCE / "llama.cpp"
WORK = REFERENCE / "gemma"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(command, log, **kwargs):
    print(" ".join(map(str, command)), flush=True)
    with log.open("w") as stream:
        subprocess.run(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT, check=True, **kwargs)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    model = REFERENCE / "models" / MANIFEST["file"]
    if not model.exists() or sha(model) != MANIFEST["sha256"]:
        raise RuntimeError(f"Missing or unverified model: {model}")
    revision = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    if revision != MANIFEST["llama_cpp"]:
        raise RuntimeError("llama.cpp revision differs from its pin")
    if subprocess.check_output(["git", "-C", str(SOURCE), "diff", "HEAD", "--", "src", "include", "ggml/src", "ggml/include"]):
        raise RuntimeError("llama.cpp inference sources must be clean")
    destination = SOURCE / "examples/dump-trace"
    destination.mkdir(exist_ok=True)
    for name in ("dump-trace.cpp", "CMakeLists.txt"):
        shutil.copy2(ROOT / "tests/acceptance/oracle" / name, destination / name)
    cmake = SOURCE / "examples/CMakeLists.txt"
    if "add_subdirectory(dump-trace)" not in cmake.read_text():
        with cmake.open("a") as stream:
            stream.write("\nadd_subdirectory(dump-trace)\n")
    run(["cmake", "-S", SOURCE, "-B", SOURCE / "build", "-DGGML_METAL=OFF", "-DGGML_BLAS=OFF",
         "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_SERVER=OFF"], WORK / "configure.log")
    run(["cmake", "--build", SOURCE / "build", "--target", "llama-quantize", "llama-dump-trace", "-j8"], WORK / "build.log")
    dequant = WORK / "gemma-3-270m-it-f32.gguf"
    run([SOURCE / "build/bin/llama-quantize", "--allow-requantize", model, dequant, "F32", "4"], WORK / "dequantize.log")
    reference_sha = sha(dequant)
    if MANIFEST.get("reference_sha256", reference_sha) != reference_sha:
        raise RuntimeError("F32-dequantized model differs from its pin")
    for name, case in MANIFEST["cases"].items():
        output = REFERENCE / "oracle" / ("gemma-" + name)
        output.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, DUMP_DIR=str(output), DUMP_N="8", DUMP_DECODE_LOGITS="1")
        for variable in ("DUMP_MESSAGES", "DUMP_USER", "DUMP_SYSTEM", "DUMP_THINKING"):
            env.pop(variable, None)
        if "messages" in case:
            messages = output / "messages.json"
            messages.write_text(json.dumps(case["messages"]) + "\n")
            env["DUMP_MESSAGES"] = str(messages)
        prompt = case.get("text", case.get("repeat", "") * case.get("count", 0) + case.get("suffix", ""))
        run([SOURCE / "build/bin/llama-dump-trace", "-m", dequant, "-p", prompt,
             "-t", "4", "-ngl", "0", "-c", "2048", "-b", "1024", "-ub", "1024", "--seed", "0"],
            output / "log.txt", env=env)
        provenance = {key: MANIFEST[key] for key in ("sha256", "revision", "llama_cpp", "dtype")}
        provenance.update(reference_sha256=reference_sha, native=False, case=case)
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("F32-dequantized SHA256:", reference_sha)


if __name__ == "__main__":
    main()
