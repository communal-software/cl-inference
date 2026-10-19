"""Generate/check offline HF SentencePiece fixtures and pinned TinyLlama expectations."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import urllib.request

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'tools/hf-sentencepiece-manifest.json'
DATA = ROOT / 'tests/local/data'
REFERENCE = ROOT / 'reference/hf-sentencepiece'


def serialized(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n'


def synthetic(original):
    result = copy.deepcopy(original)
    tokens = ['<unk>', '<s>', '</s>'] + [f'<0x{i:02X}>' for i in range(256)]
    tokens += ['▁'] + [chr(i) for i in range(33, 127)] + ['é']
    merges = ['b c', 'a b', 'h e', 'l l', 'he ll', 'hell o', '▁ hello', '▁ ▁']
    tokens += ['bc', 'ab', 'he', 'll', 'hell', 'hello', '▁hello', '▁▁', '<tool>']
    result['model']['vocab'] = {token: i for i, token in enumerate(tokens)}
    result['model']['merges'] = merges
    result['added_tokens'].append(dict(id=len(tokens)-1, content='<tool>', special=False,
                                      normalized=False, single_word=False, lstrip=False, rstrip=False))
    return result


def inputs():
    values = ['', 'Hello', 'hello', 'abc', 'ab abc bca', '  hello  ', '\t\n\r', 'a\tb\nc',
              '▁', '▁hello', 'é e\u0301', '你好 日本 العربية', '😀🦀', '\x00a\x7f',
              '<s>', '</s>', '<unk>', 'a</s>b', '<s></s><unk>', '<s> hello </s>',
              '<tool>', 'a<tool>b', '<tool><tool>', 'a  b', '123 ٤٥６']
    rng = random.Random(63)
    alphabet = 'abchello 123\t\n▁é\u0301你好😀<>/s'
    values += [''.join(rng.choices(alphabet, k=rng.randrange(1, 65))) for _ in range(80)]
    return values


def expectations(pipeline):
    tokenizer = Tokenizer.from_str(serialized(pipeline))
    cases = []
    for special in [False, True]:
        tokenizer.encode_special_tokens = not special
        for bos in [False, True]:
            for text in inputs():
                ids = tokenizer.encode(text, add_special_tokens=bos).ids
                cases.append(dict(text=text, special=special, bos=bos, ids=ids,
                                  decoded=tokenizer.decode(ids, skip_special_tokens=True),
                                  rendered=tokenizer.decode(ids, skip_special_tokens=False)))
    vocab = pipeline['model']['vocab']
    piece_cases = [[], ['▁'], ['<s>', '▁'], ['<0x61>', '<0xFF>'],
                   ['<0xE2>', '<0x96>', '<0x81>'], ['<0xF0>', '<0x9F>'],
                   ['<0xF0>', '<0x9F>', '<0x98>', '<0x80>'],
                   ['<0x61>', '<s>', '<0xFF>'], ['<0xE0>', '<0x80>', '<0x80>'],
                   ['<0xED>', '<0xA0>', '<0x80>'], ['<0xF4>', '<0x90>', '<0x80>', '<0x80>'],
                   ['<0x80>', '▁', '<0xFF>'], ['<0x61>', '▁', '<0x62>']]
    for pieces in piece_cases:
        ids = [vocab[piece] for piece in pieces]
        cases.append(dict(ids=ids, decoded=tokenizer.decode(ids, skip_special_tokens=True),
                          rendered=tokenizer.decode(ids, skip_special_tokens=False)))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    if importlib.metadata.version('tokenizers') != manifest['tokenizers']:
        raise RuntimeError('Use the pinned tokenizers version')
    artifacts = {}
    for name, digest in manifest['files'].items():
        path = REFERENCE / name
        if not path.exists() and args.download:
            url = f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{name}"
            data = urllib.request.urlopen(url).read()
            if hashlib.sha256(data).hexdigest() != digest:
                raise RuntimeError(f'Artifact checksum mismatch: {name}')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f'Artifact checksum mismatch: {name}')
        artifacts[name] = json.loads(data)
    pipeline = synthetic(artifacts['tokenizer.json'])
    outputs = {
        DATA / 'hf-sentencepiece-tokenizer.json': pipeline,
        DATA / 'hf-sentencepiece-expected.json': dict(cases=expectations(pipeline)),
        DATA / 'hf-sentencepiece-real-expected.json': dict(cases=expectations(artifacts['tokenizer.json'])),
    }
    for path, value in outputs.items():
        content = serialized(value)
        if args.check:
            if path.read_text() != content:
                raise RuntimeError(f'Expectations differ: {path}')
        else:
            path.write_text(content)
    print('HF SentencePiece reference fixtures match' if args.check else 'Generated HF SentencePiece fixtures')


if __name__ == '__main__':
    main()
