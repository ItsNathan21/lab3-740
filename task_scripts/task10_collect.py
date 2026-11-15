"""Run from /workspace in the lab container. Redirect combined stdout/stderr to results10.txt.
Optional positional stage names: A B sweep C D. B/sweep/C require the max-context build.
"""
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
import profiler as p

def cmd(*args):
    print('$', shlex.join(list(map(str,args))), flush=True)
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout, flush=True)
    result.check_returncode()
    return result.stdout

def emit(**row):
    print('TASK10_RECORD '+json.dumps(row), flush=True)

def bench(key, name, engine, params, variant=None, images=False):
    for rep in range(1,4):
        kwargs = dict(params)
        if engine=='cpu':
            kwargs.update(core=7,threads=1)
        if images and rep==1:
            kwargs['write_pgm']=True
        task=p._task(name, p.Backend(engine), dict(variant=variant,timeout=600,**kwargs))
        res=prof.run(task)
        print(res.raw, res.summary(), flush=True)
        if not res.ok or not res.compute_us_per_iter:
            raise RuntimeError(f'{key}: {res.error}')
        if images and rep==1:
            p._morph_images(task,prof.device)
        emit(key=key,rep=rep,us=res.compute_us_per_iter,rate=res.gflops,
             fp=res.fingerprint,residual=res.fields.get('residual'),temp=res.temp_c)
    time.sleep(1)

def conv(key,engine,h,c,co=None,variant=None,it=5):
    bench(key,'convolution',engine,dict(H=h,W=h,cin=c,cout=co or c,k=3,iterations=it),variant)

def build(n,kind='conv'):
    log=cmd('bash','scripts/build_student.sh',kind,'--threads',str(n))
    if 'Pushed to the board:' not in log: raise RuntimeError('No verified push')
    
def baseline():
    # Build the supplied original in isolation, leaving the working source untouched.
    with tempfile.TemporaryDirectory() as folder:
        src=Path(folder)/'student_conv.c'
        src.write_bytes(Path('kernels/lab3/conv/student_conv.c.orig').read_bytes())
        cmd('bash','scripts/build_dsp_skel.sh','kernels/convolution/dsp','--threads','1',
            '--exclude','dsp_conv_hvx_imp.c','--add',str(src),
            '--exclude','dsp_conv_morph_imp.c','--add',str(ROOT/'kernels/lab3/morph/student_morph.c'),
            '--exclude','dsp_conv_stage_imp.c','--add',str(ROOT/'kernels/lab3/vtcm/student_vtcm.c'))
        cmd('adb','push','kernels/convolution/dsp/prebuilt/libdsp_conv_skel.so','/data/local/tmp/profiler-kernels/dsp/')

stages=sys.argv[1:] or ['A','B','sweep','C','D']
prof=p.Profiler(verbose=True)
cmd('adb','devices')
hello=cmd('adb','shell',"""cd /data/local/tmp/profiler-kernels/dsp && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH='.;$PWD' ./dsp_hello_test""")
if 'A' in stages:
    cmd('python','scripts/power_pin.py','prime','4396800')
    conv('A_cpu','cpu',256,64)
    baseline()
    conv('A_1','dsp',256,64)
    for n in (2,8):
        build(n)
        conv(f'A_{n}','dsp',256,64)
if 'B' in stages:
    for prec,shapes in [('fp32',[(256,32),(256,64),(256,128),(128,128),(128,256)]),('int8',[(128,128),(128,256)])]:
        for h,c in shapes:
            for engine in ('cpu','dsp'):
                conv(f'B_{prec}_{h}_{c}_{engine}',engine,h,c,variant='int8' if prec=='int8' else None,it=2)
if 'sweep' in stages:
    for c in range(120,137):
        conv(f'sweep_{c}','dsp',128,c,128,it=2)
if 'C' in stages:
    for rep in range(1,4):
        log=cmd('bash','scripts/run_concurrent.sh')
        rows=re.findall(r'^(?:CPU \(\d+ cores\)|DSP \(HVX\))\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x',log,re.M)
        assert len(rows)==2
        fingerprints=re.findall(r'(?:CPU|DSP) solo (\S+)\s+conc (\S+)',log)
        assert len(fingerprints)==2 and all(a==b for a,b in fingerprints)
        emit(key='C',rep=rep,cpu=list(map(float,rows[0])),dsp=list(map(float,rows[1])),combined=float(re.search(r'aggregate concurrent: ([\d.]+)',log)[1]))
        time.sleep(2)
if 'D' in stages:
    # The lab image bundles the Android compiler inside the Hexagon SDK.
    if 'NDK_ROOT' not in os.environ:
        ndks = sorted(Path('/local/mnt/workspace/Qualcomm/Hexagon_SDK').glob('*/tools/android-ndk-*'))
        if ndks:
            os.environ['NDK_ROOT'] = str(ndks[-1])
    print('Morph build NDK_ROOT:', os.environ.get('NDK_ROOT', '(default)'), flush=True)
    build(8,'morph')
    for impl in ('qhl','student'):
        for op,k in [('erode',3),('dilate',3),('erode',5),('erode',7)]:
            bench(f'D_{impl}_{op}_{k}','morph','dsp',dict(impl=impl,op=op,k=k),images=k==3)
