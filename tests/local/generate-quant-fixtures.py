"""Regenerate packed block expectations with the pinned llama.cpp library."""
import ctypes
import pathlib
import struct
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parents[2]
reference = root / "reference/llama.cpp"
if subprocess.check_output(["git", "-C", str(reference), "rev-parse", "HEAD"], text=True).strip() != "8bb3cf0d216a5ae48360179f2931daf284be4d01":
    raise RuntimeError("llama.cpp does not match the pinned commit")
extension = ".dylib" if sys.platform == "darwin" else ".so"
library = ctypes.CDLL(str(reference / ("build/bin/libggml-base" + extension)))


def number(value):
    text = format(value, ".9g")
    return text.replace("e", "f") if "e" in text else text + ("f0" if "." in text else ".0f0")


def blocks(dtype):
    geometry = {"q4_0": (32, 18), "q4_k": (256, 144), "q6_k": (256, 210)}
    elements, size = geometry[dtype]
    result = bytearray()
    for index, scale in enumerate((0.0, 0.03125, -0.125, 2**-24, 0.5)):
        block = bytearray((i * 73 + index * 29 + 7) % 256 for i in range(size))
        offset = 208 if dtype == "q6_k" else 0
        block[offset:offset+2] = struct.pack("<e", scale)
        if dtype == "q4_k":
            block[2:4] = struct.pack("<e", abs(scale) * 0.5)
        result.extend(block)
    return elements * 5, result


entries = []
for dtype, suffix in (("q4_0", "q4_0"), ("q4_k", "q4_K"), ("q6_k", "q6_K")):
    count, payload = blocks(dtype)
    source = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
    target = (ctypes.c_float * count)()
    function = getattr(library, "dequantize_row_" + suffix)
    function.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)
    function(source, target, count)
    entries.append("  (:" + dtype + "\n   #(" + " ".join(map(str, payload)) + ")\n   #(" + " ".join(map(number, target)) + "))")
output = """;;; Expected values from llama.cpp 8bb3cf0, not the library decoder.
(defparameter *packed-golden-blocks*
 (mapcar (lambda (fixture)
           (list (first fixture)
                 (coerce (second fixture) '(simple-array (unsigned-byte 8) (*)))
                 (coerce (third fixture) '(simple-array single-float (*)))))
 '(""" + "\n".join(entries) + ")))\n"
(root / "tests/local/quant-golden.lisp").write_text("(in-package #:cl-inference/local/tests)\n\n" + output)
pathlib.Path("/tmp/cl-quant-golden-upstream.lisp").write_text("(in-package #:trivial-simd/tests)\n\n" + output)
