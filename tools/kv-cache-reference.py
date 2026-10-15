"""Generate pinned symmetric-Q8 cache oracles with Python MLX/MLX-LM."""
import argparse
import gc
import json
import runpy
import sys
import subprocess
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import KVCache
from mlx_lm.utils import load_model

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference/mlx"
COMMON = runpy.run_path(str(Path(__file__).with_name("mlx-reference.py")))
ALGORITHM = "symmetric-s8-f32-group32-rne-v1"
GGUF_READER_REVISION = "8bb3cf0d216a5ae48360179f2931daf284be4d01"
GGUF = {
    "llama": ("Llama-3.2-1B-Instruct-f16.gguf", "1f33ad43d2b85b908ff06fe7002b69806a57359b9b2617ca27d7bdea428ae146"),
    "qwen": ("Qwen3-0.6B-f16.gguf", "25fb880334987a71db728fb43af8568a4682208059f754a2735781f5d0970143"),
}


def encode(x):
    dim = x.shape[-1]
    padded = mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, (-dim) % 32)])
    grouped = padded.reshape(*x.shape[:-1], -1, 32)
    peak = mx.max(mx.abs(grouped), axis=-1)
    scales = mx.where(peak == 0, 1.0, mx.maximum(peak / 127.0, 1.1754943508222875e-38))
    codes = mx.clip(mx.round(grouped / scales[..., None]), -127, 127).astype(mx.int8)
    return codes.reshape(*x.shape[:-1], -1)[..., :dim], scales


def decode(pair):
    codes, scales = pair
    dim = codes.shape[-1]
    padded = mx.pad(codes, [(0, 0)] * (codes.ndim - 1) + [(0, (-dim) % 32)])
    grouped = padded.reshape(*codes.shape[:-1], -1, 32).astype(mx.float32)
    decoded = mx.clip(grouped * scales[..., None], -mx.finfo(mx.float32).max, mx.finfo(mx.float32).max)
    return decoded.reshape(*codes.shape[:-1], -1)[..., :dim]


class SymmetricCache(KVCache):
    def update_and_fetch(self, keys, values):
        def append(old, new):
            return tuple(mx.concatenate([a, b], axis=2) for a, b in zip(old, new)) if old else new

        self.keys = append(self.keys, encode(keys))
        self.values = append(self.values, encode(values))
        self.offset += keys.shape[2]
        return decode(self.keys), decode(self.values)

    @property
    def state(self):
        return (*self.keys, *self.values)


def gguf_weights(model, family, path):
    revision = subprocess.check_output(["git", "-C", str(ROOT / "reference/llama.cpp"), "rev-parse", "HEAD"], text=True).strip()
    if revision != GGUF_READER_REVISION:
        raise ValueError("GGUF reader revision differs")
    sys.path.insert(0, str(ROOT / "reference/llama.cpp/gguf-py"))
    from gguf import GGUFReader
    import numpy as np

    names = {"token_embd": "model.embed_tokens", "output_norm": "model.norm", "output": "lm_head"}
    roles = {"attn_norm": "input_layernorm", "attn_q": "self_attn.q_proj",
             "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
             "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
             "ffn_norm": "post_attention_layernorm", "ffn_gate": "mlp.gate_proj",
             "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
    tensors = GGUFReader(path).tensors
    if model.args.tie_word_embeddings:
        tied = {t.name: t.data for t in tensors if t.name in ("token_embd.weight", "output.weight")}
        if "output.weight" in tied and not np.array_equal(tied["output.weight"], tied["token_embd.weight"]):
            raise ValueError("GGUF output is not tied to embeddings")
    weights = []
    for tensor in tensors:
        parts = tensor.name.split(".")
        if parts[0] == "rope_freqs" or (parts[0] == "output" and model.args.tie_word_embeddings):
            continue
        if parts[0] == "blk":
            name = f"model.layers.{parts[1]}.{roles[parts[2]]}.weight"
        else:
            name = names[parts[0]] + ".weight"
        weights.append((name, mx.array(tensor.data).astype(mx.float32)))
    model.load_weights(weights, strict=True)
    if family == "llama":
        for layer in model.layers:
            layer.self_attn.rope.traditional = True


def generate(case, pins):
    cpu = case.startswith("cpu-")
    family = case.split("-")[1] if cpu else case.split("-")[0]
    source = f"{family}-float" if cpu else case
    entry = pins[source]
    for relative, expected in entry["files"].items():
        if COMMON["sha256"](REFERENCE / relative) != expected:
            raise ValueError(f"Reference input differs: {relative}")
    model, _ = load_model(REFERENCE / source)
    model.set_dtype(mx.float32)
    inputs = {f"reference/mlx/{name}": digest for name, digest in entry["files"].items()}
    if cpu:
        name, digest = GGUF[family]
        path = ROOT / "reference/models" / name
        if COMMON["sha256"](path) != digest:
            raise ValueError(f"GGUF input differs: {name}")
        inputs = {f"reference/models/{name}": digest}
        gguf_weights(model, family, path)
    ids = json.loads((REFERENCE / f"{source}-oracle.json").read_text())["ids"]
    cache = [SymmetricCache() for _ in model.layers]
    logits = model(mx.array([ids]), cache=cache)[:, -1, :]
    mx.eval(logits, [c.state for c in cache])
    outputs, greedy = [logits[0].tolist()], []
    for _ in range(8):
        token = int(mx.argmax(logits[0]).item())
        greedy.append(token)
        logits = model(mx.array([[token]]), cache=cache)[:, -1, :]
        mx.eval(logits, [c.state for c in cache])
        outputs.append(logits[0].tolist())
    metadata = {"algorithm": ALGORITHM, "case": case, "packages": COMMON["VERSIONS"], "inputs": inputs}
    if cpu:
        metadata["gguf_reader"] = GGUF_READER_REVISION
        metadata["source_configuration"] = entry["files"][f"{source}/config.json"]
    output = ROOT / "reference" / f"{case}-kv-q8.json"
    output.write_text(json.dumps({"metadata": metadata, "ids": ids, "logits": outputs,
                                  "greedy": greedy}, allow_nan=False) + "\n")
    print(case, greedy, flush=True)
    return {"metadata": metadata, "sha256": COMMON["sha256"](output)}


def main():
    cases = ["cpu-llama", "cpu-qwen", "llama-float", "llama-4bit", "qwen-float", "qwen-4bit"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=cases, action="append")
    args = parser.parse_args()
    if {name: version(name) for name in COMMON["VERSIONS"]} != COMMON["VERSIONS"]:
        raise ValueError("Reference package versions differ")
    pins = json.loads((ROOT / "tools/mlx-reference-manifest.json").read_text())
    manifest = ROOT / "tools/kv-cache-reference-manifest.json"
    entries = json.loads(manifest.read_text()) if manifest.exists() else {}
    for case in args.case or cases:
        entries[case] = generate(case, pins)
        manifest.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")
        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    main()
