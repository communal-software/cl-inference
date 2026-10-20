import subprocess, os, time, json
from pathlib import Path
out=Path('/tmp/inference-hotspots-2026-10-20'); out.mkdir(exist_ok=True)
def snapshot():
 return {c:subprocess.run(c.split(),capture_output=True,text=True).stdout for c in ['vm_stat','sysctl vm.swapusage','ps -axo pid,pcpu,comm']}
for model in ['QWEN-F16','QWEN-Q8']:
 for phase in ['prefill','decode']:
  stem=f'{model.lower()}-{phase}'
  before=snapshot()
  env=dict(os.environ,PROFILE_MODEL=model,PROFILE_PHASE=phase,VECLIB_MAXIMUM_THREADS='1',TRIVIAL_SIMD_BACKEND='native')
  with (out/(stem+'.log')).open('w') as log:
   p=subprocess.Popen(['sbcl','--dynamic-space-size','3072','--script','/tmp/inference-hotspots.lisp'],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
   sampled=False
   for line in p.stdout:
    log.write(line); log.flush()
    if line.startswith('READY'):
     print(stem+' sampling',flush=True)
     sampler=subprocess.Popen(['/usr/bin/sample',str(p.pid),'25','2','-file',str(out/(stem+'-native.txt'))],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
     sampled=True
   code=p.wait()
   if sampled: sampler.wait()
  (out/(stem+'-environment.json')).write_text(json.dumps({'before':before,'after':snapshot(),'exit_code':code},indent=2))
  if code: raise SystemExit(f'{stem} failed: {code}')
  print(stem+' complete',flush=True)
