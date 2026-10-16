"""Generate offline default-BPE splits and IDs from pinned llama.cpp; stdlib only."""
import argparse
import ast
import os
from pathlib import Path
import random
import shlex
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REVISION = "8bb3cf0d216a5ae48360179f2931daf284be4d01"
INPUTS = [
    "", "Hello world", "hello abc aaa héllo", "It's 'Tis I'LL we're THEY'VE I'M she'd",
    "1234567", "a1b22c333d4444", " 1234567", "123٤56789²345",
    "Hello, world!!\n\nNext", "  leading and trailing  ", "a  b   c",
    "line1\r\nline2\r\n\r\nline3  \n  x", "tab\there\t\tthere", "héllo wörld",
    "日本語のテキスト 中文", "x😀y 😀😀 !😀", "école", "price: $12.50!",
    " 12", "   ", "\n", " \n ", "...ok?!\r\nfoo", "don't stop", "'s", "x'", "'",
    "foo\u00a0bar", "a\u2003b", "a\u3000b", "C++/Java_script-ish", "snake_case_name",
    "a\n \nb", "text   ", "  \n  ", "٣٤ ²³", "a\u2028b", "$+<=>^~|`©😀_a",
    "1234567890", "١٢٣٤٥٦٧٨٩", "123٤123456", "a\t b", "a\x00b", "a\x1cb",
    "\\\"!#$%&'()*+,-./:;<=>?@[\\]^_`{|}~", "👩\u200d💻 café\u0301", "𐐀𐐨 𝟘𝟙𝟚𝟛",
    "<s>hello</s><think>", "\r\n\t\r\n a", "abc 123hello_abc",
]


def run(args, **kwargs):
    return subprocess.run(args, check=True, capture_output=True, **kwargs).stdout


def byte_chars():
    direct = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    extra = iter(range(256, 512))
    return [chr(b if b in direct else next(extra)) for b in range(256)]


def gguf(path, pre):
    chars = byte_chars()
    tokens = chars + ["he", "ll", "hell", "hello", "ab", "bc", "aa", chars[195] + chars[169],
                      "<s>", "</s>", "<think>"]
    merges = ["h e", "l l", "he ll", "hell o", "b c", "a b", "a a", chars[195] + " " + chars[169]]
    metadata = [
        ("general.architecture", 8, "llama"), ("tokenizer.ggml.model", 8, "gpt2"),
        ("tokenizer.ggml.tokens", 9, (8, tokens)),
        ("tokenizer.ggml.token_type", 9, (5, [1] * 264 + [3, 3, 4])),
        ("tokenizer.ggml.merges", 9, (8, merges)),
        ("tokenizer.ggml.bos_token_id", 4, 264), ("tokenizer.ggml.eos_token_id", 4, 265),
        ("tokenizer.ggml.add_bos_token", 7, False),
    ]
    if pre is not None:
        metadata.append(("tokenizer.ggml.pre", 8, pre))

    def value(kind, item):
        if kind == 8:
            data = item.encode("utf-8")
            return struct.pack("<Q", len(data)) + data
        if kind == 9:
            subtype, items = item
            return struct.pack("<IQ", subtype, len(items)) + b"".join(value(subtype, x) for x in items)
        return struct.pack({4: "<I", 5: "<i", 7: "<?"}[kind], item)

    data = struct.pack("<IIQQ", 0x46554747, 3, 0, len(metadata))
    for key, kind, item in metadata:
        data += value(8, key) + struct.pack("<I", kind) + value(kind, item)
    path.write_bytes(data + bytes((-len(data)) % 32))


def lisp_string(text):
    escaped = ""
    for char in text:
        escaped += {"~": "~~", "\n": "~n", "\r": "~r", "\t": "~t", '"': '\\"', "\\": "\\\\"}.get(
            char, char if 32 <= ord(char) <= 126 else "~u{%x}" % ord(char))
    return '"' + escaped + '"'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "reference/llama.cpp")
    parser.add_argument("--check", action="store_true", help="Verify tracked expectations without writing")
    args = parser.parse_args()
    ref = args.reference.resolve()
    revision = run(["git", "-C", str(ref), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION or run(["git", "-C", str(ref), "diff", "HEAD", "--", "src/unicode.cpp",
                                   "src/unicode-data.cpp", "src/unicode.h", "src/unicode-data.h",
                                   "src/llama-vocab.cpp"]):
        raise RuntimeError("Reference tokenizer sources must match " + REVISION)
    tokenizer = ref / "build/bin/llama-tokenize"
    version = subprocess.run([str(tokenizer), "--version"], check=True, capture_output=True, text=True)
    if REVISION[:7] not in version.stdout + version.stderr:
        raise RuntimeError("Rebuild llama-tokenize at " + REVISION)
    rng = random.Random(32)
    inputs = INPUTS + ["".join(rng.choice("ab 'S_+$`123٤²\t\n\r😀é\u0301") for _ in range(rng.randrange(1, 65)))
                       for _ in range(100)]
    rows = []
    with tempfile.TemporaryDirectory(prefix="default-tokenizer-") as directory:
        tmp = Path(directory)
        splitter = tmp / "split"
        run(shlex.split(os.environ.get("CXX", "c++")) + ["-std=c++17", "-O2", "-I" + str(ref / "src"),
            str(ROOT / "tools/default-tokenizer-reference.cpp"), str(ref / "src/unicode.cpp"),
            str(ref / "src/unicode-data.cpp"), "-o", str(splitter)])
        models = []
        for i, pre in enumerate((None, "", "default")):
            model = tmp / f"vocab-{i}.gguf"
            gguf(model, pre)
            models.append(model)
        for text in inputs:
            pieces = [bytes.fromhex(line).decode("utf-8") for line in
                      run([str(splitter)], input=text.encode()).decode().splitlines()]
            assert "".join(pieces) == text
            ids = []
            for model in models:
                output = run([str(tokenizer), "-m", str(model), "--stdin", "--ids",
                              "--no-bos", "--no-parse-special", "--no-escape", "--offline"], input=text.encode()).decode()
                ids.append(ast.literal_eval(output.strip()))
            assert ids[0] == ids[1] == ids[2], (text, ids)
            rows.append("    (" + lisp_string(text) + " (" + " ".join(map(lisp_string, pieces)) + ") (" +
                        " ".join(map(str, ids[0])) + "))")
    content = ('(in-package #:cl-inference/local/tests)\n\n'
               ';;; Generated by tools/default-tokenizer-reference.py; llama.cpp ' + REVISION + '.\n'
               '(defparameter *default-bpe-reference*\n  \'(\n' + "\n".join(rows) + "))\n")
    target = ROOT / "tests/local/default-tokenizer-expected.lisp"
    if args.check:
        if target.read_text() != content:
            raise RuntimeError("Default tokenizer expectations differ")
    else:
        target.write_text(content)
    print(f"Verified {len(rows)} split cases and {len(rows) * 3} tokenizer cases")


if __name__ == "__main__":
    main()
