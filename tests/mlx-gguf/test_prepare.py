"""Offline tests for selective reference generation."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    'prepare', Path(__file__).resolve().parents[2] / 'tools/prepare-mlx-gguf.py')
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class PrepareTests(unittest.TestCase):
    def test_family_selection_preserves_unselected_references(self):
        for families in ([], ['llama'], ['qwen'], ['gemma'], ['llama', 'qwen']):
            with self.subTest(families=families), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'tools').mkdir()
                (root / 'tests/acceptance/oracle').mkdir(parents=True)
                (root / 'tests/acceptance/oracle/dump-trace.cpp').write_text('oracle')
                models = {family + '-q4_0': {'file': family}
                          for family in ('llama', 'qwen')}
                (root / 'tests/acceptance/quant-models.json').write_text(json.dumps(models))
                (root / 'tests/acceptance/gemma-model.json').write_text(
                    json.dumps({'cases': {'plain': {}}}))
                previous = {}
                for family in ('llama', 'qwen', 'gemma'):
                    modes = ['plain'] if family == 'gemma' else ['plain', 'chat']
                    if family == 'qwen':
                        modes.append('thinking')
                    for mode in modes:
                        case = f'{family}-{mode}' if family == 'gemma' else f'{family}-q4_0-{mode}'
                        previous[case] = {'unselected': case}
                        output = root / 'reference/oracle' / case
                        output.mkdir(parents=True)
                        for name in ('prompt.ids', 'greedy.ids', 'trace.txt', 'top.txt',
                                     'provenance.json', 'decode.f32'):
                            (output / name).write_text(name)
                manifest = root / 'tools/mlx-gguf-reference-manifest.json'
                manifest.write_text(json.dumps(previous))
                args = [arg for family in families for arg in ('--family', family)]
                selected = set(families or ('llama', 'qwen', 'gemma'))
                with patch.object(prepare, 'ROOT', root), \
                     patch.object(prepare.subprocess, 'run') as run, \
                     patch.dict(os.environ, {'DUMP_MESSAGES': 'unwanted'}):
                    prepare.main(args)
                actual = json.loads(manifest.read_text())
                for case in previous:
                    if case.split('-')[0] in selected:
                        self.assertEqual(actual[case]['oracle_source_sha256'],
                                         prepare.sha(root / 'tests/acceptance/oracle/dump-trace.cpp'))
                        self.assertEqual(len(actual[case]['files']), 6)
                    else:
                        self.assertEqual(actual[case], previous[case])
                commands = [call.args[0] for call in run.call_args_list]
                self.assertEqual(len(commands), bool(selected & {'llama', 'qwen'})
                                 + ('gemma' in selected))
                for call in run.call_args_list:
                    self.assertNotIn('DUMP_MESSAGES', call.kwargs['env'])
                    self.assertEqual(call.kwargs['env']['DUMP_DECODE_LOGITS'], '1')
                if selected & {'llama', 'qwen'}:
                    self.assertEqual(commands[0][2:], [arg for family in sorted(
                        selected & {'llama', 'qwen'}) for arg in ('--family', family)])


if __name__ == '__main__':
    unittest.main()
