"""Regenerate offline Lisp fixtures from the hash-pinned Jinja template matrix."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / 'tools/chat-template-manifest.json').read_text())
if jinja2.__version__ != MANIFEST['jinja2']:
    raise RuntimeError(f"Use Jinja2 {MANIFEST['jinja2']}")
if version('MarkupSafe') != MANIFEST['markupsafe']:
    raise RuntimeError(f"Use MarkupSafe {MANIFEST['markupsafe']}")


def fail(message):
    raise ValueError(message)


env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
env.filters['tojson'] = lambda x, **kw: json.dumps(x, ensure_ascii=False, **kw)
env.globals['raise_exception'] = fail
WEATHER = {'type': 'function', 'function': {
    'name': 'get_weather', 'parameters': {'type': 'object', 'properties': {
        'city': {'type': 'string', 'description': 'City name'},
        'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']},
        'days': {'type': 'integer', 'minimum': 1, 'maximum': 7}},
        'required': ['city'], 'additionalProperties': False}, 'description': 'Get the current weather'}}
TYPED = {'type': 'function', 'function': {
    'name': 'typed', 'parameters': {'type': 'object', 'properties': {
        'tags': {'type': 'array', 'items': {'type': 'string'}},
        'metadata': {'type': 'object', 'additionalProperties': {'type': 'string'}},
        'details': {'type': 'object', 'properties': {'count': {'type': 'integer'}}, 'required': [], 'additionalProperties': False},
        'enabled': {'type': 'boolean'}, 'anything': {}}, 'required': [], 'additionalProperties': False}, 'description': 'Typed parameters'}}
NULLABLE = {'type': 'function', 'function': {
    'name': 'nullable', 'parameters': {'type': 'object', 'properties': {'note': {'type': ['string', 'null']}},
                                      'required': [], 'additionalProperties': False}, 'description': 'Nullable parameter'}}
SYSTEM = {'role': 'system', 'content': 'You are a helpful assistant.'}
USER = {'role': 'user', 'content': 'What is the weather in Paris?'}


def assistant(content, calls=None):
    result = {'role': 'assistant', 'content': content}
    if calls:
        result['tool_calls'] = [{'type': 'function', 'function': {'name': 'get_weather', 'arguments': x}} for x in calls]
    return result


def cases():
    yield 'plain', [SYSTEM, USER], {}
    yield 'no-system', [USER], {}
    yield 'no-generation', [USER], {'add_generation_prompt': False}
    yield 'multi-turn', [SYSTEM, USER, assistant('Sunny.'), USER], {}
    yield 'empty', [], {}
    yield 'empty-text', [{'role': 'user', 'content': ''}], {}
    yield 'whitespace', [{'role': 'system', 'content': '  Be brief.  \n'}, {'role': 'user', 'content': '\n héllo 日本  '}], {}
    yield 'tools', [SYSTEM, USER], {'tools': [WEATHER]}
    yield 'typed-tools', [USER], {'tools': [TYPED]}
    yield 'nullable-tools', [USER], {'tools': [NULLABLE]}
    yield 'tools-no-system', [USER], {'tools': [WEATHER]}
    yield 'call', [USER, assistant('', [{'city': 'Paris', 'days': 3}])], {'tools': [WEATHER]}
    yield 'call-text', [USER, assistant('Checking.', [{'city': 'Paris'}])], {'tools': [WEATHER]}
    yield 'two-calls', [USER, assistant('', [{'city': 'Paris'}, {'city': 'Rome'}])], {'tools': [WEATHER]}
    yield 'result', [USER, assistant('', [{'city': 'Paris'}]), {'role': 'tool', 'content': ' sunny '}], {'tools': [WEATHER]}
    yield 'two-results', [USER, assistant('', [{'city': 'Paris'}]), {'role': 'tool', 'content': 'sunny'}, {'role': 'tool', 'content': 'warm'}, assistant('Done.')], {'tools': [WEATHER]}
    yield 'escaping', [USER, assistant('', [{'city': 'Zürich "CH"\n', 'unit': 'celsius'}]), {'role': 'tool', 'content': 'say "hi"\nthere \\ ✓'}], {'tools': [WEATHER]}
    yield 'thinking-off', [SYSTEM, USER], {'enable_thinking': False}
    yield 'reasoning-history', [USER, assistant('<think>\nsecret\n</think>\n\nSunny.'), USER], {}
    yield 'wrapped-result-user', [USER, assistant('<think>secret</think>call'), {'role': 'user', 'content': '  <tool_response>sunny</tool_response>  '}, assistant('<think>after</think>answer')], {}
    yield 'reasoning-last', [USER, assistant('<think>\nsecret\n</think>\n\nSunny.')], {}
    yield 'system-tools', [SYSTEM, USER], {'tools': [WEATHER], 'tools_in_user_message': False}
    yield 'builtin', [USER, assistant('', [{'city': 'Paris'}])], {'tools': [WEATHER], 'builtin_tools': ['get_weather']}


def string(value):
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


def lisp(value):
    if value is True:
        return 't'
    if value is False or value is None:
        return 'nil'
    if isinstance(value, str):
        return string(value)
    if isinstance(value, list):
        return '(' + ' '.join(lisp(x) for x in value) + ')'
    if isinstance(value, dict):
        pairs = []
        for key, child in value.items():
            if key == 'tool_calls':
                calls = []
                for call in child:
                    f = call['function']
                    arguments = []
                    for k, v in f['arguments'].items():
                        arguments.extend([':' + k, ':' + v if k == 'unit' else lisp(v)])
                    calls.append('(:name :' + f['name'] + ' :arguments (' + ' '.join(arguments) + '))')
                pairs.extend([':tool-calls', '(' + ' '.join(calls) + ')'])
            else:
                pairs.extend([':' + key.replace('_', '-'), ':' + child if key == 'role' else lisp(child)])
        return '(' + ' '.join(pairs) + ')'
    return str(value)


out = ['(in-package #:cl-inference/local/tests)', '', ';; Generated by tools/chat-template-reference.py.', '(defparameter *family-reference-cases*', "  '("]
for entry in MANIFEST['templates']:
    path = ROOT / 'tests/local/templates' / entry['file']
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise ValueError(f'Hash mismatch: {path}')
    template = env.from_string(data.decode())
    for name, messages, options in cases():
        cls = entry['class']
        if name in ('system-tools', 'builtin') and cls != 'llama31-chat':
            continue
        if cls == 'llama30-chat' and 'tools' in options:
            continue
        if cls == 'r1-llama-chat' and any('tool_calls' in m for m in messages):
            # This template takes a JSON string and a null content for tool messages.
            messages = json.loads(json.dumps(messages))
            for m in messages:
                if 'tool_calls' in m:
                    m['content'] = None
                    for call in m['tool_calls']:
                        call['function']['arguments'] = json.dumps(call['function']['arguments'], ensure_ascii=False)
        params = dict(bos_token='', add_generation_prompt=True, date_string='26 Jul 2024', enable_thinking=True)
        params.update(options)
        try:
            result = string(template.render(messages=messages, **params))
        except (ValueError, jinja2.UndefinedError, TypeError, RecursionError):
            result = ':error'
        # Lisp uses structured arguments for both R1 templates.
        if cls == 'r1-llama-chat':
            for m in messages:
                for call in m.get('tool_calls', []):
                    call['function']['arguments'] = json.loads(call['function']['arguments'])
        keys = [':date', string('26 Jul 2024'), ':add-generation-prompt', lisp(params['add_generation_prompt'])]
        if 'tools' in options:
            tool = options['tools'][0]
            keys.extend([':tools', ':typed' if tool is TYPED else ':nullable' if tool is NULLABLE else ':weather'])
        if 'enable_thinking' in options:
            keys.extend([':enable-thinking', lisp(options['enable_thinking'])])
        if 'tools_in_user_message' in options:
            keys.extend([':tools-in-user-message', lisp(options['tools_in_user_message'])])
        if 'builtin_tools' in options:
            keys.extend([':builtin-tools', '(:get_weather)'])
        out.append(f"    (local:{cls} {string(entry['file'] + '/' + name)} {lisp(messages)} ({' '.join(keys)}) {result})")
out.append('))')
(ROOT / 'tests/local/chat-families-expected.lisp').write_text('\n'.join(out) + '\n')
