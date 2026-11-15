"""Parse a complete Task 10 transcript. Usage: python task_scripts/task10_report.py."""
import collections
import json
from pathlib import Path
import re
import statistics as st

root=Path(__file__).resolve().parents[1]
text=(root/'task_output/results10.txt').read_text()
data=collections.defaultdict(dict)
for line in text.splitlines():
    if line.startswith('TASK10_RECORD '):
        r=json.loads(line.partition(' ')[2])
        data[r['key']][r['rep']]=r
def vals(key,field):
    assert set(data[key])=={1,2,3},key
    return [row[field] for row in data[key].values()]
def mu(key,field='rate'):
    return st.mean(vals(key,field))
def cov(v):
    return 100*st.stdev(v)/st.mean(v)
def disp(values,places=2):
    return f'{st.mean(values):.{places}f} ({cov(values):.2f}\\%)'
lines=['% Requires graphicx. Upload the two named PNG files beside this .tex file.',
       r'\noindent All reported means use three runs. Parentheses give sample CoV.',
       r'FP32 rates are GFLOP/s. INT8 rates are GOP/s. For the standalone convolution runs, the profiler uses average CPU time per iteration and the last on-device DSP iteration time.']
def section(name):
    lines.append(r'\par\medskip\noindent\textbf{Task '+name+r'}\par')
def table(head,rows):
    lines.extend([r'\begin{center}\small\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{l'+'r'*(len(head)-1)+'}',r'\hline',
                  ' & '.join(head)+r' \\ \hline'])
    lines.extend(' & '.join(map(str,row))+r' \\' for row in rows)
    lines.extend([r'\hline\end{tabular}',r'\end{center}'])
section('10.1')
hw=re.search(r'HVX units:.*?\((\d+) x (\d+)B',text)
vt=re.search(r'VTCM total:\s*(\d+) KB',text)
lines.append(r'FP32, $256\times256$, $C_{in}=C_{out}=64$, $K=3$, five iterations per process. Prime pinned at 4.3968 GHz. Rates below are GFLOP/s.\par')
lines.append(f'HVX contexts: {hw[1]} at {hw[2]} bytes each. VTCM: {vt[1]} KiB.')
rows=[]
for tag,label,n in [('A_cpu','Prime (1 core)',0),('A_1','DSP baseline',1),('A_2','DSP, 2 contexts',2),('A_8','DSP, 8 contexts',8)]:
    fp=vals(tag,'fp')
    assert len(set(fp))==1 and (not n or fp==vals('A_1','fp'))
    speed=mu(tag)/mu('A_1')
    rows.append([label,*[f'{v:.2f}' for v in vals(tag,'rate')],disp(vals(tag,'rate')),f'{fp[0]:.2f}',f'{speed:.2f}' if n else '--',f'{100*speed/n:.1f}\\%' if n else '--'])
table(['Engine','Run 1','Run 2','Run 3','Mean (CoV)','FP','Speedup','Eff.'],rows)
section('10.2')
lines.append(r'One Prime core and one DSP using the maximum-context build; $K=3$, two iterations per process.')
rows=[]
for prec,shapes in [('fp32',[(256,32),(256,64),(256,128),(128,128),(128,256)]),('int8',[(128,128),(128,256)])]:
    for h,c in shapes:
        tag=f'B_{prec}_{h}_{c}_'
        cpu,dsp=tag+'cpu',tag+'dsp'
        rows.append([prec.upper(),f'${h}\\times{h}$',c,disp(vals(cpu,'rate')),disp(vals(dsp,'rate')),f'{mu(dsp)/mu(cpu):.3f}'])
table(['Type','Spatial size',r'$C_{in}=C_{out}$','Prime rate','DSP rate','DSP/Prime'],rows)
section('10.3')
lines.append(r'Shape measurements appear in Task 10.2.\par')
section('10.4')
lines.append(r'Model inputs from handout: HVX 1024 bits at approximately 1 GHz, NEON 128 bits at approximately 4.4 GHz.')
section('10.5')
table([r'$C_{in}$','DSP GFLOP/s (CoV)','DSP time ($\\mu$s)'],[[c,disp(vals(f'sweep_{c}','rate')),f'{mu(f"sweep_{c}","us"):.1f}'] for c in range(120,137)])
lines.append(r'FP32, $128\times128$, $C_{out}=128$, $K=3$.')
section('10.6')
lines.append(r'All eight CPU cores and the DSP, $256\times256$, $C_{in}=C_{out}=64$. Rates are wall-time GFLOP/s. The slowdown factor $S$ is solo throughput divided by joint throughput.')
vs=[data['C'][i] for i in (1,2,3)]
rows=[]
for i,r in enumerate(vs,1):
    a,b,s=r['cpu']; d,e,t=r['dsp']
    rows.append([i,a,b,d,e,r['combined'],s,t])
table(['Run','CPU solo','CPU joint','DSP solo','DSP joint','Joint sum',r'$S_{CPU}$',r'$S_{DSP}$'],rows)
rows=[]
for metric,series in [
    ('CPU solo GFLOP/s', [r['cpu'][0] for r in vs]),
    ('CPU joint GFLOP/s', [r['cpu'][1] for r in vs]),
    ('DSP solo GFLOP/s', [r['dsp'][0] for r in vs]),
    ('DSP joint GFLOP/s', [r['dsp'][1] for r in vs]),
    ('Joint sum GFLOP/s', [r['combined'] for r in vs]),
    ('CPU slowdown factor', [r['cpu'][2] for r in vs]),
    ('DSP slowdown factor', [r['dsp'][2] for r in vs]),
]:
    rows.append([metric,disp(series),f'{min(series):.2f}--{max(series):.2f}'])
table(['Metric','Mean (CoV)','Range'],rows)
section('10.7')
lines.append(r'One-core measurements appear in Task 10.2 and eight-core measurements in Task 10.6.\par')
section('10.8')
lines.append(r'Solo and joint rates appear in Task 10.6.\par')
section('10.9')
rows=[]
for op in ('erode','dilate'):
    entries=[]
    for impl in ('qhl','student'):
        a=vals(f'D_{impl}_{op}_3','residual')
        assert len(set(a))==1
        entries.append(int(a[0]))
    assert entries[0]==entries[1],(op,entries)
    rows.append([op,*entries])
table(['Operation ($K=3$)','QHL residual (pixels)','Student residual (pixels)'],rows)
for op in ('erode','dilate'):
    lines.extend([r'\begin{figure}[htbp]\centering',
                  r'\includegraphics[width=\linewidth]{qhl_'+op+'_salt_k3_compare.png}',
                  r'\caption{QHL '+op+r', $K=3$. Left to right: clean, damaged, repaired.}\end{figure}'])
section('10.10')
lines.append(r'Student and QHL residuals for both filters appear in Task 10.9.\par')
section('10.11')
rows=[]
for k in (3,5,7):
    a,b=f'D_student_erode_{k}',f'D_qhl_erode_{k}'
    rows.append([k,disp(vals(a,'us')),disp(vals(b,'us')),f'{mu(a,"us")/mu(b,"us"):.2f}',f'{2*1024*768/(mu(b,"us")*1000):.3f}'])
table(['Window','Student time ($\\mu$s)','QHL time ($\\mu$s)','QHL speedup','QHL useful GB/s'],rows)
section('10.12')
lines.append(r'The last column of Task 10.11 uses $B=2WH/t$ for QHL erosion. ')
lines.append(r'Erosion on $1024\times768$ 8-bit pixels. Useful bytes per pass: $2(1024)(768)$.')
output=root/'task_output/task10_framework.tex'
output.write_text('\n'.join(lines)+'\n')
print(output)
