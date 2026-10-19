"""Generate cached-decode GGUF references and pin their hashes for MLX acceptance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('llama', 'qwen', 'gemma'), action='append')
    families = set(parser.parse_args(argv).family or ('llama', 'qwen', 'gemma'))
    env = dict(os.environ, DUMP_DECODE_LOGITS='1')
    for name in ('DUMP_MESSAGES', 'DUMP_USER', 'DUMP_SYSTEM', 'DUMP_THINKING'):
        env.pop(name, None)
    quant_families = sorted(families & {'llama', 'qwen'})
    if quant_families:
        command = ['python3', str(ROOT / 'tests/acceptance/prepare-quant.py')]
        for family in quant_families:
            command.extend(['--family', family])
        subprocess.run(command, env=env, check=True)
    if 'gemma' in families:
        subprocess.run(['python3', str(ROOT / 'tests/acceptance/prepare-gemma.py')],
                       env=env, check=True)
    models = json.loads((ROOT / 'tests/acceptance/quant-models.json').read_text())
    manifest_path = ROOT / 'tools/mlx-gguf-reference-manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    source = ROOT / 'tests/acceptance/oracle/dump-trace.cpp'
    for name, entry in models.items():
        if name.split('-')[0] not in families:
            continue
        for mode in ['plain', 'chat'] + (['thinking'] if name.startswith('qwen') else []):
            case = f'{name}-{mode}'
            directory = ROOT / 'reference/oracle' / case
            files = {file: sha(directory / file) for file in
                     ['prompt.ids', 'greedy.ids', 'trace.txt', 'top.txt', 'provenance.json', 'decode.f32']}
            manifest[case] = {'model': entry, 'oracle_source_sha256': sha(source), 'files': files}
    if 'gemma' in families:
        gemma = json.loads((ROOT / 'tests/acceptance/gemma-model.json').read_text())
        for mode in gemma['cases']:
            case = f'gemma-{mode}'
            directory = ROOT / 'reference/oracle' / case
            files = {file: sha(directory / file) for file in
                     ['prompt.ids', 'greedy.ids', 'trace.txt', 'top.txt', 'provenance.json', 'decode.f32']}
            manifest[case] = {'model': gemma, 'oracle_source_sha256': sha(source), 'files': files}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
