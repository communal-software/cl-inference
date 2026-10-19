"""Prepare and verify the pinned Gemma HF tokenizer and F32 inference references."""
import argparse
import copy
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'tools/gemma-hf-manifest.json'
REFERENCE = ROOT / 'reference/gemma-hf'
DATA = ROOT / 'tests/local/data'
VERSIONS = {'torch': '2.10.0', 'transformers': '5.5.0', 'tokenizers': '0.22.2'}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def serialized(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n'


def synthetic(original):
    pipeline = copy.deepcopy(original)
    tokens = ['<pad>', '<eos>', '<bos>', '<unk>'] + [f'<0x{i:02X}>' for i in range(256)]
    tokens += ['▁'] + [chr(i) for i in range(33, 127)] + ['é']
    merges = ['b c', 'a b', 'h e', 'l l', 'he ll', 'hell o', '▁ hello', '▁ ▁']
    tokens += ['bc', 'ab', 'he', 'll', 'hell', 'hello', '▁hello', '▁▁', '<tool>', '\n', '\n\n']
    pipeline['model']['vocab'] = {token: i for i, token in enumerate(tokens)}
    pipeline['model']['merges'] = merges
    pipeline['added_tokens'] = [dict(id=i, content=token, special=i < 4,
                                    normalized=False, single_word=False, lstrip=False, rstrip=False)
                                for i, token in enumerate(tokens) if i < 4 or token in ['<tool>', '\n', '\n\n']]
    return pipeline


def expectations(pipeline):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_str(serialized(pipeline))
    texts = ['', 'Hello', 'hello', 'abc', 'ab abc bca', '  hello  ', '\t\n\r', 'a\tb\nc',
             '▁', '▁hello', 'é e\u0301', '你好 日本 العربية', '😀🦀', '\x00a\x7f',
             '<bos>', '<eos>', '<unk>', 'a<eos>b', '<bos><eos><unk>', '<bos> hello <eos>',
             '<tool>', 'a<tool>b', '<tool><tool>', 'a  b', '123 ٤٥６', '\n' * 20,
             '<start_of_turn>user\nhello<end_of_turn>\n', '<unused0>', '</s>']
    rng = random.Random(74)
    texts += [''.join(rng.choices('abchello 123\t\n▁é\u0301你好😀<>/s', k=rng.randrange(1, 65))) for _ in range(80)]
    cases = []
    for special in [False, True]:
        tokenizer.encode_special_tokens = not special
        for bos in [False, True]:
            for text in texts:
                ids = tokenizer.encode(text, add_special_tokens=bos).ids
                cases.append(dict(text=text, special=special, bos=bos, ids=ids,
                                  decoded=tokenizer.decode(ids), rendered=tokenizer.decode(ids, skip_special_tokens=False)))
    vocab = pipeline['model']['vocab']
    for pieces in [[], ['▁'], ['<bos>', '▁'], ['<0x61>', '<0xFF>'],
                   ['<0xE2>', '<0x96>', '<0x81>'], ['<0xF0>', '<0x9F>'],
                   ['<0xF0>', '<0x9F>', '<0x98>', '<0x80>'], ['<0x61>', '<bos>', '<0xFF>'],
                   ['<0xE0>', '<0x80>', '<0x80>'], ['<0xED>', '<0xA0>', '<0x80>'],
                   ['<0xF4>', '<0x90>', '<0x80>', '<0x80>'], ['<0x80>', '▁', '<0xFF>']]:
        ids = [vocab[piece] for piece in pieces]
        cases.append(dict(ids=ids, decoded=tokenizer.decode(ids), rendered=tokenizer.decode(ids, skip_special_tokens=False)))
    return dict(cases=cases)


def inference():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    root = REFERENCE / 'model'
    model = AutoModelForCausalLM.from_pretrained(root, local_files_only=True, dtype=torch.float32,
                                               attn_implementation='eager').eval()
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    cases = json.loads((ROOT / 'tests/acceptance/gemma-model.json').read_text())['cases']
    output = {'packages': VERSIONS, 'precision': 'float32', 'cases': {}}
    with torch.inference_mode():
        for name, case in cases.items():
            if 'messages' in case:
                prompt = tokenizer.apply_chat_template(case['messages'], tokenize=False, add_generation_prompt=True)
                ids = tokenizer.encode(prompt, add_special_tokens=False)
            else:
                prompt = case.get('text', case.get('repeat', '') * case.get('count', 0) + case.get('suffix', ''))
                ids = tokenizer.encode(prompt)
            layers = []
            def capture(module, args, result):
                hidden = result[0] if isinstance(result, tuple) else result
                layers.append(hidden[0, -1].float().tolist())
            hooks = [layer.register_forward_hook(capture) for layer in model.model.layers]
            result = model(torch.tensor([ids]), use_cache=True, logits_to_keep=1)
            for hook in hooks:
                hook.remove()
            greedy = []
            path = REFERENCE / f'{name}.f32'
            with path.open('wb') as stream:
                for _ in range(9):
                    logits = result.logits[0, -1].float()
                    stream.write(logits.numpy().astype('<f4').tobytes())
                    if len(greedy) == 8:
                        break
                    token = int(logits.argmax())
                    greedy.append(token)
                    if token in [1, 106]:
                        break
                    result = model(torch.tensor([[token]]), past_key_values=result.past_key_values,
                                   use_cache=True, logits_to_keep=1)
            output['cases'][name] = dict(prompt=prompt, ids=ids, layers=layers, greedy=greedy,
                                         rows=path.stat().st_size // (4 * model.config.vocab_size))
            print(name, len(ids), greedy, flush=True)
    (REFERENCE / 'oracle.json').write_text(serialized(output))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if {name: version(name) for name in VERSIONS} != VERSIONS:
        raise RuntimeError('Use the pinned reference requirements')
    manifest = json.loads(MANIFEST.read_text())
    for name, digest in manifest['files'].items():
        path = REFERENCE / 'model' / name
        if not path.exists() and args.download:
            token = os.environ.get('HF_TOKEN')
            token_path = Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface')) / 'token'
            if not token and token_path.exists():
                token = token_path.read_text().strip()
            headers = {'Authorization': f'Bearer {token}'} if token else {}
            request = urllib.request.Request(f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{name}", headers=headers)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + '.part')
            with urllib.request.urlopen(request) as source, temporary.open('wb') as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
            if sha(temporary) != digest:
                raise RuntimeError(f'Artifact checksum mismatch: {name}')
            temporary.replace(path)
        if sha(path) != digest:
            raise RuntimeError(f'Artifact checksum mismatch: {name}')
    original = json.loads((REFERENCE / 'model/tokenizer.json').read_text())
    fixture = synthetic(original)
    for path, value in [(DATA / 'gemma-hf-tokenizer.json', fixture),
                        (DATA / 'gemma-hf-expected.json', expectations(fixture)),
                        (DATA / 'gemma-hf-real-expected.json', expectations(original))]:
        content = serialized(value)
        if args.check:
            if path.read_text() != content:
                raise RuntimeError(f'Fixture differs: {path}')
        else:
            path.write_text(content)
    if args.inference:
        inference()
        outputs = ['oracle.json'] + [f'{name}.f32' for name in json.loads((REFERENCE / 'oracle.json').read_text())['cases']]
        hashes = {name: sha(REFERENCE / name) for name in outputs}
        if args.check:
            if hashes != manifest['outputs']:
                raise RuntimeError('Inference reference differs')
        else:
            manifest['packages'] = VERSIONS
            manifest['outputs'] = hashes
            manifest['generator_sha256'] = sha(Path(__file__))
            MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
    elif 'outputs' in manifest:
        for name, digest in manifest['outputs'].items():
            if sha(REFERENCE / name) != digest:
                raise RuntimeError(f'Reference checksum mismatch: {name}')
    print('Gemma HF references verified', flush=True)


if __name__ == '__main__':
    main()
