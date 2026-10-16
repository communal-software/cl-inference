"""Generate cached-decode GGUF references and pin their hashes for MLX acceptance."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    subprocess.run(['python3', str(ROOT / 'tests/acceptance/prepare-quant.py')],
                   env=dict(os.environ, DUMP_DECODE_LOGITS='1'), check=True)
    models = json.loads((ROOT / 'tests/acceptance/quant-models.json').read_text())
    manifest = {}
    source = ROOT / 'tests/acceptance/oracle/dump-trace.cpp'
    for name, entry in models.items():
        for mode in ['plain', 'chat'] + (['thinking'] if name.startswith('qwen') else []):
            case = f'{name}-{mode}'
            directory = ROOT / 'reference/oracle' / case
            files = {file: sha(directory / file) for file in
                     ['prompt.ids', 'greedy.ids', 'trace.txt', 'top.txt', 'provenance.json', 'decode.f32']}
            manifest[case] = {'model': entry, 'oracle_source_sha256': sha(source), 'files': files}
    (ROOT / 'tools/mlx-gguf-reference-manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
