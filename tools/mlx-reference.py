import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model
from transformers import AutoTokenizer


MODELS = {
    "llama": ("meta-llama/Llama-3.2-1B-Instruct", "9213176726f574b556790deb65791e0c5aa438b6"),
    "qwen3": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
}
VERSIONS = {"mlx": "0.32.2", "mlx-lm": "0.32.0", "transformers": "5.19.0", "tokenizers": "0.23.3"}
DATE = "26 Jul 2024"
SIDECARS = ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
            "special_tokens_map.json", "chat_template.jinja")


def input_files(root):
    files = [root / name for name in SIDECARS if (root / name).is_file()]
    index = root / "model.safetensors.index.json"
    if index.is_file():
        files.append(index)
        shards = set(json.loads(index.read_text())["weight_map"].values())
        files.extend(root / name for name in sorted(shards))
    else:
        files.append(root / "model.safetensors")
    for file in files:
        if not file.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Input escapes model directory: {file}")
    return files


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, help="Update a reviewed reference manifest after generating the oracle")
    args = parser.parse_args()
    packages = {name: version(name) for name in VERSIONS}
    if packages != VERSIONS:
        raise ValueError(f"Reference dependency versions differ: {packages}")
    root = args.model.resolve()
    if args.output.resolve() != root.parent / f"{root.name}-oracle.json":
        raise ValueError("Oracle must be named <model-directory>-oracle.json beside the model directory")
    model, config = load_model(root)
    architecture = config["model_type"]
    repository, revision = MODELS[architecture]
    quantization = config.get("quantization", config.get("quantization_config"))
    if quantization and (quantization.get("bits") != 4 or quantization.get("group_size") != 64
                         or quantization.get("mode", "affine") != "affine"):
        raise ValueError("Reference quantization must be affine 4-bit with group size 64")
    model.set_dtype(mx.float32)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, date_string=DATE)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    cache = make_prompt_cache(model)
    logits = model(mx.array([ids]), cache=cache)[:, -1, :]
    mx.eval(logits, [entry.state for entry in cache])
    initial = logits[0].tolist()
    greedy, cached = [], []
    for _ in range(8):
        token = int(mx.argmax(logits[0]).item())
        greedy.append(token)
        logits = model(mx.array([[token]]), cache=cache)[:, -1, :]
        mx.eval(logits, [entry.state for entry in cache])
        cached.append(logits[0].tolist())
    texts = ["hello", "é", "e\u0301", "日本語 😀 1234567"]
    texts += (["<|begin_of_text|>", "<|start_header_id|>assistant<|end_header_id|>\n\n", "<|eot_id|>"]
              if architecture == "llama" else ["<|im_start|>assistant\n", "<|im_end|>"])
    encodings = [{"text": text, "add_bos": bos,
                  "ids": tokenizer.encode(text, add_special_tokens=bos)}
                 for text in texts for bos in ([False, True] if architecture == "llama" else [False])]
    metadata = {"repository": repository, "revision": revision, "architecture": architecture,
                "packages": packages, "precision": "float32", "quantization": quantization,
                "date": DATE}
    data = {"metadata": metadata, "prompt": prompt, "ids": ids, "logits": initial,
            "cached_logits": cached, "greedy": greedy, "encodings": encodings}
    args.output.write_text(json.dumps(data, allow_nan=False) + "\n")
    if args.manifest:
        manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else {}
        files = input_files(root) + [args.output.resolve()]
        hashes = {str(file.relative_to(root.parent)): sha256(file) for file in files}
        manifest[root.name] = {"metadata": metadata, "files": hashes}
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(root.name, greedy, tokenizer.decode(greedy))


if __name__ == "__main__":
    main()
