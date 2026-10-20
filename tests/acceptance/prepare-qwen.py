"""Download pinned Qwen3 F16 and prepare three CPU llama.cpp references."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
REVISION = "8bb3cf0d216a5ae48360179f2931daf284be4d01"
MODEL = "Qwen3-0.6B-f16.gguf"
MODEL_SHA256 = "25fb880334987a71db728fb43af8568a4682208059f754a2735781f5d0970143"
MODEL_URL = (
    "https://huggingface.co/second-state/Qwen3-0.6B-GGUF/resolve/"
    "9c1e90ceb7ee4dfe02235b861aa08f0640f89459/" + MODEL
)
CASES = {"qwen-f16": None, "qwen-f16-thinking": "1", "qwen-f16-chat": "0"}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_model(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(".gguf.partial")
        print(f"Downloading {MODEL_URL}", flush=True)
        urllib.request.urlretrieve(MODEL_URL, partial)
        if sha256(partial) != MODEL_SHA256:
            raise RuntimeError("Downloaded Qwen3 model SHA256 differs from its pin")
        partial.replace(path)
    if sha256(path) != MODEL_SHA256:
        raise RuntimeError("Qwen3 model SHA256 differs from its pin")


def run(command, log, **kwargs):
    print(" ".join(map(str, command)), flush=True)
    with log.open("w") as stream:
        subprocess.run(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT,
                       check=True, **kwargs)


def prepare(root=ROOT):
    reference = root / "reference"
    work = reference / "qwen-build"
    work.mkdir(parents=True, exist_ok=True)
    model = reference / "models" / MODEL
    download_model(model)
    source = reference / "llama.cpp"
    if not source.exists():
        run(["git", "clone", "--depth", "1", "--branch", "b11574",
             "https://github.com/ggml-org/llama.cpp.git", source], work / "clone.log")
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise RuntimeError("llama.cpp revision differs from its pin")
    if subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--",
                                "src", "include", "common", "ggml/src", "ggml/include"]):
        raise RuntimeError("llama.cpp inference sources must be clean")
    destination = source / "examples/dump-trace"
    destination.mkdir(parents=True, exist_ok=True)
    oracle_source = root / "tests/acceptance/oracle"
    for name in ("dump-trace.cpp", "CMakeLists.txt"):
        shutil.copy2(oracle_source / name, destination / name)
    cmake = source / "examples/CMakeLists.txt"
    if "add_subdirectory(dump-trace)" not in cmake.read_text():
        with cmake.open("a") as stream:
            stream.write("\nadd_subdirectory(dump-trace)\n")
    run(["cmake", "-S", source, "-B", work / "build", "-DGGML_METAL=OFF",
         "-DGGML_CUDA=OFF", "-DGGML_BLAS=OFF", "-DLLAMA_CURL=OFF",
         "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_SERVER=OFF", "-DCMAKE_BUILD_TYPE=Release"],
        work / "configure.log")
    run(["cmake", "--build", work / "build", "--target", "llama-dump-trace", "--parallel", "2"],
        work / "build.log")
    for case, thinking in CASES.items():
        output = reference / "oracle" / case
        output.mkdir(parents=True, exist_ok=True)
        env = {key: value for key, value in os.environ.items() if not key.startswith("DUMP_")}
        env.update(DUMP_DIR=str(output), DUMP_N="8")
        if thinking is not None:
            env.update(DUMP_SYSTEM="Answer in one word.", DUMP_USER="What is the capital of France?",
                       DUMP_THINKING=thinking)
        prompt = "The capital of France is" if thinking is None else ""
        run([work / "build/bin/llama-dump-trace", "-m", model, "-p", prompt,
             "-t", "2", "-ngl", "0", "-c", "512", "--seed", "0"], output / "log.txt", env=env)
        files = ("prompt.ids", "greedy.ids", "trace.txt", "top.txt")
        provenance = {
            "model_url": MODEL_URL, "sha256": MODEL_SHA256, "llama_cpp": REVISION,
            "oracle_source_sha256": sha256(oracle_source / "dump-trace.cpp"),
            "case": case, "prompt": prompt, "thinking": thinking,
            "files": {name: sha256(output / name) for name in files},
        }
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    prepare()
