"""Offline regression tests for the CPU CI reference preparation."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "prepare_qwen", Path(__file__).with_name("prepare-qwen.py"))
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class DownloadTests(unittest.TestCase):
    def test_download_is_verified_before_installation_and_existing_models_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / prepare.MODEL

            def download(url, destination):
                self.assertEqual(url, prepare.MODEL_URL)
                destination.write_bytes(b"model")

            with patch.object(prepare.urllib.request, "urlretrieve", side_effect=download) as fetch, \
                 patch.object(prepare, "MODEL_SHA256", prepare.hashlib.sha256(b"model").hexdigest()):
                prepare.download_model(model)
                prepare.download_model(model)
                self.assertEqual(fetch.call_count, 1)
            self.assertEqual(model.read_bytes(), b"model")
            with self.assertRaisesRegex(RuntimeError, "SHA256"):
                prepare.download_model(model)

    def test_bad_download_is_not_installed(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / prepare.MODEL
            with patch.object(prepare.urllib.request, "urlretrieve",
                              side_effect=lambda url, path: path.write_bytes(b"wrong")):
                with self.assertRaisesRegex(RuntimeError, "SHA256"):
                    prepare.download_model(model)
            self.assertFalse(model.exists())


class PreparationTests(unittest.TestCase):
    def fixture(self, root):
        (root / "reference/llama.cpp/examples").mkdir(parents=True)
        (root / "reference/llama.cpp/examples/CMakeLists.txt").write_text("# examples\n")
        oracle = root / "tests/acceptance/oracle"
        oracle.mkdir(parents=True)
        for name in ("dump-trace.cpp", "CMakeLists.txt"):
            (oracle / name).write_text(name)

    def test_revision_and_dirty_sources_are_rejected_before_build(self):
        for output, message in ((["wrong\n"], "revision"),
                                ([prepare.REVISION + "\n", b"dirty"], "clean")):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.fixture(root)
                with patch.object(prepare, "download_model"), \
                     patch.object(prepare.subprocess, "check_output", side_effect=output), \
                     patch.object(prepare, "run") as run:
                    with self.assertRaisesRegex(RuntimeError, message):
                        prepare.prepare(root)
                    run.assert_not_called()

    def test_all_cases_have_isolated_environment_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)

            def run(command, log, **kwargs):
                if "env" in kwargs:
                    output = Path(kwargs["env"]["DUMP_DIR"])
                    for name in ("prompt.ids", "greedy.ids", "trace.txt", "top.txt"):
                        (output / name).write_text(name)

            with patch.object(prepare, "download_model"), \
                 patch.object(prepare.subprocess, "check_output",
                              side_effect=[prepare.REVISION + "\n", b""]), \
                 patch.object(prepare, "run", side_effect=run) as commands, \
                 patch.dict(os.environ, {"DUMP_MESSAGES": "stale", "DUMP_N": "100",
                                         "DUMP_DECODE_LOGITS": "1"}):
                prepare.prepare(root)
            self.assertEqual(len(commands.call_args_list), 5)
            self.assertIn("-DGGML_METAL=OFF", commands.call_args_list[0].args[0])
            cases = commands.call_args_list[2:]
            self.assertEqual([call.kwargs["env"].get("DUMP_THINKING") for call in cases],
                             [None, "1", "0"])
            for call in cases:
                env = call.kwargs["env"]
                self.assertEqual(env["DUMP_N"], "8")
                self.assertNotIn("DUMP_MESSAGES", env)
                self.assertNotIn("DUMP_DECODE_LOGITS", env)
                output = Path(env["DUMP_DIR"])
                provenance = json.loads((output / "provenance.json").read_text())
                self.assertEqual(provenance["case"], output.name)
                self.assertEqual(provenance["sha256"], prepare.MODEL_SHA256)
                self.assertEqual(provenance["llama_cpp"], prepare.REVISION)
                self.assertEqual(len(provenance["files"]), 4)
            self.assertEqual((root / "reference/llama.cpp/examples/dump-trace/dump-trace.cpp").read_text(),
                             "dump-trace.cpp")

    def test_subprocess_failure_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(prepare.subprocess, "run",
                              side_effect=subprocess.CalledProcessError(1, "cmake")):
                with self.assertRaises(subprocess.CalledProcessError):
                    prepare.run(["cmake"], Path(directory) / "failure.log")


if __name__ == "__main__":
    unittest.main()
