import argparse, json
from pathlib import Path
import mlx.core as mx
from mlx_lm.utils import load_model
from mlx_lm.models.cache import make_prompt_cache
from transformers import AutoTokenizer

p=argparse.ArgumentParser()
p.add_argument('model')
p.add_argument('output')
a=p.parse_args()
root=Path(a.model)
model,config=load_model(root)
model.set_dtype(mx.float32)
tokenizer=AutoTokenizer.from_pretrained(root)
messages=[{'role':'user','content':'What is the capital of France?'}]
prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
ids=tokenizer.encode(prompt,add_special_tokens=False)
cache=make_prompt_cache(model)
logits=model(mx.array([ids]),cache=cache)[:,-1,:]
mx.eval(logits,[c.state for c in cache])
initial=logits[0].tolist()
greedy=[]
for i in range(8):
 token=int(mx.argmax(logits[0]).item())
 greedy.append(token)
 logits=model(mx.array([[token]]),cache=cache)[:,-1,:]
 mx.eval(logits,[c.state for c in cache])
texts=['hello','é','e\u0301','日本語 😀 1234567','<|im_start|>assistant\n']
data={'prompt':prompt,'ids':ids,'logits':initial,'greedy':greedy,'encodings':[[s,tokenizer.encode(s,add_special_tokens=False)] for s in texts]}
Path(a.output).write_text(json.dumps(data))
print(root, greedy,tokenizer.decode(greedy))
