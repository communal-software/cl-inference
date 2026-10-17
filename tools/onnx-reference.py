"""Prepare pinned assets and independent ONNX Runtime/tokenizer acceptance data."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import urllib.request
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'tests/onnx/model.json'
MODEL=ROOT/'reference/onnx/smollm2'

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--download',action='store_true')
 args=parser.parse_args()
 manifest=json.loads(MANIFEST.read_text())
 for name,expected in manifest['files'].items():
  path=MODEL/name
  if not path.exists() and args.download:
   path.parent.mkdir(parents=True,exist_ok=True)
   urllib.request.urlretrieve(f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{name}",path)
  if not path.is_file() or digest(path)!=expected: raise RuntimeError(f'Missing or hash-mismatched asset: {path}')
 if ort.__version__!=manifest['runtime']: raise RuntimeError('Reference requires the pinned runtime')
 tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 options=ort.SessionOptions();options.intra_op_num_threads=1
 runtime=ort.InferenceSession(str(MODEL/'onnx/model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
 config=json.loads((MODEL/'config.json').read_text())
 prompts=[
  [dict(role='user',content='Reply with a single word: hello.')],
  [dict(role='system',content='Answer briefly.'),dict(role='user',content='What is 2 + 2?')],
  [dict(role='user',content='Hello!'),dict(role='assistant',content='Hi!'),dict(role='user',content='Say café in French.')],
 ]
 cases=[]
 for messages in prompts:
  ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
  past={f'past_key_values.{i}.{part}':np.zeros((1,config['num_key_value_heads'],0,config['head_dim']),dtype=np.float32)
        for i in range(config['num_hidden_layers']) for part in ('key','value')}
  def forward(tokens,start):
   nonlocal past
   feeds=dict(input_ids=np.array([tokens],dtype=np.int64),attention_mask=np.ones((1,start+len(tokens)),dtype=np.int64),
              position_ids=np.arange(start,start+len(tokens),dtype=np.int64)[None,:],**past)
   values=runtime.run(None,feeds); outputs=dict(zip([x.name for x in runtime.get_outputs()],values))
   past={name.replace('present.','past_key_values.'):value for name,value in outputs.items() if name!='logits'}
   return outputs['logits'][0,-1]
  logits=forward(ids,0);initial=logits.tolist();generated=[];decode_logits=None
  for i in range(8):
   token=int(np.argmax(logits))
   if token==config['eos_token_id']: break
   generated.append(token);logits=forward([token],len(ids)+i)
   if len(generated)==2: decode_logits=logits.tolist()
  cases.append(dict(messages=messages,prompt_ids=ids,initial_logits=initial,cached_logits=decode_logits,
                    generated_ids=generated,text=tokenizer.decode(generated,skip_special_tokens=False)))
  print('Reference',repr(cases[-1]['text']),flush=True)
 texts=['','Hello world!','123 ٤٥６','é e\u0301 你好 👋','  leading\ntrailing  ',"I'm I'M a's a'S",'<|im_start|>',
        'a\x04b','ÀÁõÿ']
 texts += [chr(i) for i in range(32)]
 token_cases=[dict(text=text,ids=tokenizer.encode(text,add_special_tokens=False),
                   decoded=tokenizer.decode(tokenizer.encode(text,add_special_tokens=False),skip_special_tokens=False)) for text in texts]
 result=dict(manifest_sha256=digest(MANIFEST),versions={name:importlib.metadata.version(name) for name in ('onnxruntime','transformers','tokenizers','numpy','jinja2')},
             cases=cases,tokenizer=token_cases)
 output=MODEL.parent/'oracle.json';output.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n')
 (MODEL.parent/'verified.json').write_text(json.dumps(dict(oracle_sha256=digest(output),manifest_sha256=digest(MANIFEST)))+'\n')
 print(output,digest(output))

if __name__=='__main__':main()
