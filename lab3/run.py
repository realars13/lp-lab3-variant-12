from copy import deepcopy
from time import perf_counter
import numpy as np
from common import *

def build_model(d, formulation='compact', relax=False, large_m=False):
    a=np.array(d['coverage']);m,n=a.shape; pairs=list(zip(*np.nonzero(a)))
    nv=n+m+(len(pairs) if formulation=='aggregate' else 0)
    c=np.zeros(nv);c[n:n+m]=-np.array(d['weights']);rows=[];lo=[];hi=[]
    row=np.zeros(nv);row[:n]=d['costs'];rows.append(row);lo.append(-np.inf);hi.append(d['budget'])
    if formulation=='compact':
        for i in range(m):
            row=np.zeros(nv);row[:n]=-a[i];row[n+i]=1
            rows.append(row);lo.append(-np.inf);hi.append(0)
    else:
        for i in range(m):
            row=np.zeros(nv);row[n+i]=-1
            for k,(ii,j) in enumerate(pairs):
                if ii==i:row[n+m+k]=1
            rows.append(row);lo.append(0);hi.append(0)
        for j in range(n):
            row=np.zeros(nv);row[j]=-(1000 if large_m else int(a[:,j].sum()))
            for k,(i,jj) in enumerate(pairs):
                if jj==j:row[n+m+k]=1
            rows.append(row);lo.append(-np.inf);hi.append(0)
    return matrix_model(c,rows,lo,hi,np.ones(nv),[] if relax else range(n+m))

def verify(d,x,formulation='compact',integer=True,large_m=False):
    a=np.array(d['coverage']);m,n=a.shape;y=np.array(x[:n]);z=np.array(x[n:n+m])
    violations=[0.,float(max(-np.array(x))),float(max(np.array(x)-1)),float(np.array(d['costs'])@y-d['budget'])]
    link_error=0.
    if formulation=='compact':violations.append(float(max(z-a@y)))
    else:
        pairs=list(zip(*np.nonzero(a)));assigned=np.zeros(m);load=np.zeros(n)
        for val,(i,j) in zip(x[n+m:],pairs):assigned[i]+=val;load[j]+=val
        M=np.full(n,1000) if large_m else a.sum(axis=0)
        link_error=max(0.,float(max(load-M*y)))
        violations.extend([float(max(abs(assigned-z))),link_error])
    if integer:
        violations.append(float(max(abs(np.r_[y,z]-np.round(np.r_[y,z])))))
        # Независимый предметный контроль: покрытая точка действительно видна выбранной камере.
        covered=(a[:,y>.5].sum(axis=1)>0) if np.any(y>.5) else np.zeros(m,bool)
        violations.append(float(max(z-covered)))
    error=max(violations)
    return {'ok':error<=TOL,'max_violation':error,'objective':-float(np.array(d['weights'])@z),
            'budget_used':float(np.array(d['costs'])@y),'budget_slack':float(d['budget']-np.array(d['costs'])@y),
            'big_m_violation':link_error,'integer_checked':integer}

def greedy(d):
    start=perf_counter();a=np.array(d['coverage'],bool);w=np.array(d['weights']);c=np.array(d['costs'])
    y=np.zeros(a.shape[1]);covered=np.zeros(a.shape[0],bool);budget=d['budget']
    while True:
        choices=[]
        for j in range(len(c)):
            if not y[j] and c[j]<=budget:
                gain=w[a[:,j]&~covered].sum()
                if gain>0:choices.append((gain/c[j],gain,-j,j))
        if not choices:break
        j=max(choices)[3];y[j]=1;budget-=c[j];covered|=a[:,j]
    return np.r_[y,covered.astype(float)],perf_counter()-start

def run():
    d=load_data('lab3');out=ROOT/'lab3/results';out.mkdir(exist_ok=True)
    scenarios=[];runs={};checks={};solution=[];scdata={}
    for name,factor in [('base',1.),('deficit',.8),('stress',.6)]:
        data=deepcopy(d);data['budget']=int(np.floor(d['budget']*factor));scdata[name]=data
        r=solve(build_model(data),out/f'logs/{name}_milp.log');lp=solve(build_model(data,relax=True),out/f'logs/{name}_lp.log')
        assert r['status']=='Optimal' and lp['status']=='Optimal'
        checks[name]=finish_check(verify(data,r['x']),r['objective'])
        checks[name+'_lp']=finish_check(verify(data,lp['x'],integer=False),lp['objective'])
        assert matrix_check(build_model(data),r['x'])['ok']
        xg,tg=greedy(data);cg=verify(data,xg);assert cg['ok']; checks[name+'_greedy']=cg
        n=len(data['costs']);m=len(data['weights']); chosen=[f'C{j+1}' for j in range(n) if r['x'][j]>.5]
        covered=[f'P{i+1}' for i in range(m) if r['x'][n+i]>.5]
        scenarios.append({'scenario':name,'budget':data['budget'],'status':r['status'],'utility':-r['objective'],
            'lp_bound':-lp['objective'],'integrality_gap_percent':100*(r['objective']-lp['objective'])/max(1,abs(r['objective'])),
            'mip_gap':r['mip_gap'],'cameras':','.join(chosen),'covered_points':','.join(covered),'seconds':r['seconds'],
            'greedy_utility':-cg['objective'],'greedy_gap_percent':100*(cg['objective']-r['objective'])/max(1,abs(r['objective'])),
            'greedy_seconds':tg})
        runs[name]={'milp':r,'lp':lp,'greedy_x':xg}
        for kind,sol in [('MILP',r['x']),('LP',lp['x']),('greedy',xg)]:
            for j,value in enumerate(sol):solution.append({'scenario':name,'method':kind,'variable':f'C{j+1}' if j<n else f'P{j-n+1}','value':value})
    # Две эквивалентные MILP-формулировки и отдельный опыт с чрезмерным M.
    formulations=[];formulation_runs={};formchecks={}
    # Дефицитный бюджет выявляет различие релаксаций: базовый допускает полное покрытие.
    formdata=deepcopy(d);formdata['budget']=scdata['deficit']['budget']
    for label,form,large in [('compact','compact',False),('aggregate','aggregate',False),('large_M','aggregate',True)]:
        lr=solve(build_model(formdata,form,True,large),out/f'logs/form_{label}_lp.log')
        samples=[];search_samples=[];rr=None
        for rep in range(3):
            rr=solve(build_model(formdata,form,False,large),out/f'logs/form_{label}_milp_{rep}.log')
            samples.append(rr['seconds']);assert rr['status']=='Optimal'
            if rr['solve_phase_seconds'] is not None:search_samples.append(rr['solve_phase_seconds'])
            finish_check(verify(formdata,rr['x'],form,True,large),rr['objective'])
        formchecks[label]=finish_check(verify(formdata,rr['x'],form,True,large),rr['objective'])
        formchecks[label+'_lp']=finish_check(verify(formdata,lr['x'],form,False,large),lr['objective'])
        assert abs(rr['objective']-runs['deficit']['milp']['objective'])<=TOL
        formulation_runs[label]={'milp':rr,'lp':lr}
        formulations.append({'formulation':label,'variables':rr['variables'],'constraints':rr['constraints'],
            'lp_bound':-lr['objective'],'milp_utility':-rr['objective'],'integrality_gap_percent':100*(rr['objective']-lr['objective'])/abs(rr['objective']),
            'seconds_median':float(np.median(samples)),'nodes':rr['nodes'],'mip_gap':rr['mip_gap'],
            'solve_phase_seconds_median':float(np.median(search_samples)) if search_samples else None,
            'budget':formdata['budget'],'lp_spending':formchecks[label+'_lp']['budget_used'],
            'lp_sum_cameras':float(sum(lr['x'][:len(d['costs'])]))})
    scale=[];scalechecks={};scaleruns={}
    for n in [10,20,40]:
        data=json.loads((ROOT/f'lab3/scale_{n}.json').read_text(encoding='utf-8'))
        lp=solve(build_model(data,relax=True),out/f'logs/scale_{n}_lp.log')
        rr=solve(build_model(data),out/f'logs/scale_{n}_milp.log')
        assert rr['status']=='Optimal'
        scalechecks[str(n)]=finish_check(verify(data,rr['x']),rr['objective'])
        scalechecks[str(n)+'_lp']=finish_check(verify(data,lp['x'],integer=False),lp['objective'])
        scaleruns[str(n)]={'milp':rr,'lp':lp}
        scale.append({'cameras':n,'points':len(data['weights']),'seconds':rr['seconds'],'mip_gap_percent':100*rr['mip_gap'],
           'integrality_gap_percent':100*(rr['objective']-lp['objective'])/abs(rr['objective']),'nodes':rr['nodes'],'utility':-rr['objective'],'lp_bound':-lp['objective']})
    csv_write(out/'scenarios.csv',scenarios);csv_write(out/'solution.csv',solution)
    csv_write(out/'formulations.csv',formulations);csv_write(out/'scale.csv',scale)
    save_json(out/'verification.json',checks);save_json(out/'solver_results.json',runs);save_json(out/'scenario_data.json',scdata)
    save_json(out/'formulation_results.json',formulation_runs);save_json(out/'formulation_verification.json',formchecks)
    save_json(out/'formulation_data.json',formdata)
    save_json(out/'scale_results.json',scaleruns);save_json(out/'scale_verification.json',scalechecks)
    for label,rs in formulation_runs.items():
        for mode,result in rs.items():csv_write(out/f'form_{label}_{mode}_solution.csv',[{'index':j,'value':x} for j,x in enumerate(result['x'])])
    bad=np.array(runs['base']['milp']['x']);bad[0]=.4;assert not verify(d,bad)['ok']
    bad=np.ones(len(bad));assert not verify(d,bad)['ok']
    weak=formulation_runs['aggregate']['milp']['x'];bad=np.array(weak);bad[:len(d['costs'])]=0
    assert not verify(d,bad,'aggregate')['ok']
    fig,ax=plt.subplots(1,2,figsize=(10,4));ax[0].plot([s['cameras'] for s in scale],[s['seconds']*1000 for s in scale],'-o')
    ax[0].set_ylabel('Время решения, мс')
    ax[1].plot([s['cameras'] for s in scale],[s['integrality_gap_percent'] for s in scale],'-o',label='LP gap')
    ax[1].plot([s['cameras'] for s in scale],[s['mip_gap_percent'] for s in scale],'-s',label='B&B gap');ax[1].set_ylabel('Зазор, %');ax[1].legend()
    for aa in ax:aa.set_xlabel('Кандидаты камер');aa.grid(alpha=.3)
    figure_save(out/'scaling.png')
    p=np.array(d['points']);c=np.array(d['cameras']);n=len(c);x=runs['base']['milp']['x'];selected=np.array(x[:n])>.5;covered=np.array(x[n:])>.5
    fig,ax=plt.subplots(figsize=(8,6))
    ax.scatter(p[covered,0],p[covered,1],c='#2d8760',label='Покрытые точки');ax.scatter(p[~covered,0],p[~covered,1],c='#bd4f4f',label='Непокрытые точки')
    ax.scatter(c[:,0],c[:,1],marker='^',facecolors='none',edgecolors='#777',label='Кандидаты')
    ax.scatter(c[selected,0],c[selected,1],marker='^',s=100,c='#174c80',label='Выбранные камеры')
    # Близкие метки разносятся в противоположные стороны.
    for coords,prefix,default in [(p,'P',(4,3)),(c,'C',(4,-9))]:
        for i,xy in enumerate(coords):
            neighbours=np.flatnonzero(np.linalg.norm(coords-xy,axis=1)<.35)
            if len(neighbours)>1:
                offset=(-10,-12) if i==min(neighbours) else (10,3)
            else:offset=default
            ax.annotate(f'{prefix}{i+1}',xy,xytext=offset,textcoords='offset points',fontsize=7,
                        ha='right' if offset[0]<0 else 'left')
    ax.set_xlabel('x, условные единицы');ax.set_ylabel('y, условные единицы');ax.legend(loc='upper left',bbox_to_anchor=(1,1));figure_save(out/'cameras.png')
    return {'data':d,'runs':runs,'scenarios':scenarios,'checks':checks,'formulations':formulations,'formulation_runs':formulation_runs,'scale':scale}
