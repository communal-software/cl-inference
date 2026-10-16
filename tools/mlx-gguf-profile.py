"""Measure complete MLX GGUF calls in isolated Apple Silicon processes."""
import argparse
import json
from pathlib import Path
import platform
from datetime import datetime, timezone
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--tokens', type=int, default=64)
    parser.add_argument('--batch', type=int, default=64)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    models = json.loads((ROOT / 'tests/acceptance/quant-models.json').read_text())
    metadata = {'recorded_at': datetime.now(timezone.utc).isoformat(),
                'lisp': subprocess.check_output(['sbcl', '--version'], text=True).strip(),
                'hardware': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                'os': platform.platform(), 'tokens': args.tokens, 'batch': args.batch, 'cases': {}}
    metadata['dependencies'] = {}
    for name in ['cl-mlx', 'cl-gpu']:
        repository = ROOT.parent / name
        if (repository / '.git').exists():
            metadata['dependencies'][name] = {
                'commit': subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip(),
                'dirty': bool(subprocess.check_output(['git', '-C', str(repository), 'status', '--porcelain'], text=True))}
    for name, pin in models.items():
        output = args.output / f'{name}.csv'
        command = ['/usr/bin/time', '-l', 'sbcl', '--dynamic-space-size', '8GB', '--script',
                   str(ROOT / 'tools/mlx-gguf-bench.lisp'), str(ROOT / 'reference/models' / pin['file']),
                   str(output.resolve()), str(args.tokens), str(args.batch)]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        (args.output / f'{name}.log').write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f'{name} failed; see its log')
        peak_rss = next(int(line.split()[0]) for line in result.stderr.splitlines()
                        if 'maximum resident set size' in line)
        metadata['cases'][name] = {'model': pin, 'peak_rss_bytes': peak_rss}
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(name, 'peak RSS', peak_rss, flush=True)


if __name__ == '__main__':
    main()
