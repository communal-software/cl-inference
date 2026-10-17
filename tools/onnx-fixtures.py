"""Regenerate tiny decoder fixtures with development-only onnx==1.20.1."""
from pathlib import Path
import json
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper as nh
root=Path(__file__).resolve().parents[1]/'tests/onnx/fixtures'
root.mkdir(exist_ok=True,parents=True)
bs=list(range(33,127))+list(range(161,173))+list(range(174,256)); cs=bs[:]; n=0
for b in range(256):
 if b not in bs: bs.append(b);cs.append(256+n);n+=1
vocab={chr(c):b for b,c in zip(bs,cs)}
vocab.update({'<|endoftext|>':256,'<|im_start|>':257,'<|im_end|>':258})
config=dict(model_type='llama',num_hidden_layers=1,num_attention_heads=1,num_key_value_heads=1,hidden_size=2,head_dim=2,
 vocab_size=259,max_position_embeddings=512,eos_token_id=258)
tokenizer=dict(model=dict(type='BPE',vocab=vocab,merges=[],unk_token=None),normalizer=None,
 pre_tokenizer=dict(type='Sequence',pretokenizers=[dict(type='Digits',individual_digits=True),dict(type='ByteLevel',add_prefix_space=False,use_regex=True)]),
 post_processor=None,decoder=dict(type='ByteLevel'),added_tokens=[dict(id=i,content=t,special=True) for t,i in vocab.items() if i>=256])
real=json.loads((root.parents[2]/'reference/onnx/smollm2/tokenizer_config.json').read_text())
sidecar={k:real[k] for k in ('chat_template','bos_token','eos_token','unk_token','pad_token')}
for name,obj in [('config.json',config),('tokenizer.json',tokenizer),('tokenizer_config.json',sidecar)]:
 (root/name).write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True)+'\n')
info=h.make_tensor_value_info
inputs=[info(x,T.INT64,[1,'sequence']) for x in ('input_ids','attention_mask','position_ids')]
inputs += [info('past_key_values.0.'+p,T.FLOAT,[1,1,'past',2]) for p in ('key','value')]
outputs=[info('logits',T.FLOAT,[1,'sequence',259])]+[info('present.0.'+p,T.FLOAT,[1,1,'total',2]) for p in ('key','value')]
table=np.full((259,259),-10,dtype=np.float32);table[:,111]=10;table[111,:]=-10;table[111,107]=10;table[107,:]=-10;table[107,258]=10
init=[nh.from_array(table,'table'),h.make_tensor('layout',T.INT64,[4],[1,1,-1,1]),h.make_tensor('repeat',T.INT64,[4],[1,1,1,2])]
nodes=[h.make_node('Gather',['table','input_ids'],['scores']),h.make_node('Cast',['position_ids'],['positions'],to=T.FLOAT),
 h.make_node('Reshape',['positions','layout'],['rows']),h.make_node('Tile',['rows','repeat'],['new'])]
nodes += [h.make_node('Concat',['past_key_values.0.'+p,'new'],['present.0.'+p],axis=2) for p in ('key','value')]
for slow in (False,True):
 ns=nodes[:];initializers=init[:]
 if slow:
  body=h.make_graph([h.make_node('Identity',['condition'],['continue']),h.make_node('Add',['state','one'],['next'])],
   'count',[info('iteration',T.INT64,[]),info('condition',T.BOOL,[]),info('state',T.INT64,[])],
   [info('continue',T.BOOL,[]),info('next',T.INT64,[])],[h.make_tensor('one',T.INT64,[],[1])])
  initializers += [h.make_tensor('billion',T.INT64,[],[1000000000]),h.make_tensor('yes',T.BOOL,[],[True]),h.make_tensor('zero',T.INT64,[],[0]),h.make_tensor('threshold',T.INT64,[],[100]),h.make_tensor('three',T.INT64,[],[3]),h.make_tensor('nilfloat',T.FLOAT,[],[0])]
  ns += [h.make_node('Size',['input_ids'],['size']),h.make_node('Greater',['size','threshold'],['large']),h.make_node('Cast',['large'],['largeint'],to=T.INT64),h.make_node('Mul',['largeint','billion'],['many']),h.make_node('Add',['many','three'],['iterations']),
   h.make_node('Loop',['iterations','yes','zero'],['total'],body=body),h.make_node('Cast',['total'],['totalfloat'],to=T.FLOAT),
   h.make_node('Mul',['totalfloat','nilfloat'],['offset']),h.make_node('Add',['scores','offset'],['logits'])]
 else: ns += [h.make_node('Identity',['scores'],['logits'])]
 m=h.make_model(h.make_graph(ns,'tiny-decoder',inputs,outputs,initializers),opset_imports=[h.make_opsetid('',14)],ir_version=7)
 onnx.checker.check_model(m);onnx.save(m,root/('slow.onnx' if slow else 'model.onnx'))
