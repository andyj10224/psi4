import sys, json
from pathlib import Path
import numpy as np
import lambda_probe as native

def fixture(seed=17,no=3,nv=4,naux=6,truncated=True,local_fit=False,fov=False,local_pao=False):
    rng=np.random.default_rng(seed)
    raw=rng.normal(scale=.1,size=(naux,no+nv,no+nv));raw=(raw+raw.swapaxes(1,2))/2
    a=rng.normal(size=(naux,naux));metric=a@a.T+np.eye(naux)
    pairs=[(i,j) for i in range(no) for j in range(no)]
    index=np.arange(no*no).reshape(no,no);rev=[index[j,i] for i,j in pairs]
    frames={};auxs={};fits={};paos={}
    orbital_paos=[set(range(nv))-{i%nv} if local_pao else set(range(nv)) for i in range(no)]
    for i,j in pairs:
        if i>j:continue
        parent=sorted(orbital_paos[i]|orbital_paos[j]);size=len(parent)
        n=size if not truncated else 1+(i+j*2)%max(1,size-1)
        x=np.zeros((nv,n));x[parent]=np.linalg.qr(rng.normal(size=(size,size)))[0][:,:n]
        aux=list(range(naux)) if not local_fit else sorted(rng.choice(naux,size=3+(i+j)%(naux-2),replace=False))
        v,u=np.linalg.eigh(metric[np.ix_(aux,aux)]);fit=np.einsum('pq,qij->pij',(u*v**-.5)@u.T,raw[aux])
        for p in {(i,j),(j,i)}:frames[p]=x;auxs[p]=aux;fits[p]=fit;paos[p]=parent
    d={'ij_to_i_j_':pairs,'ij_to_ji_':rev,'n_pno_':[frames[p].shape[1] for p in pairs],
       'i_j_to_ij_':index.tolist(),'i_j_to_ij_strong_':index.tolist(),
       'lmopair_to_lmos_':[list(range(no)) for p in pairs],'lmopair_to_lmos_dense_':[list(range(no)) for p in pairs],
       'lmopair_to_paos_':[paos[p] for p in pairs],'lmopair_to_ribfs_':[auxs[p] for p in pairs],
       'F_lmo_':np.diag(np.linspace(-1.3,-.5,no)),'F_pao_':np.diag(np.linspace(.2,1.2,nv)),
       'F_lmo_pao_':rng.normal(scale=.01,size=(no,nv)) if fov else np.zeros((no,nv)), 'S_pao_':np.eye(nv),
       'X_pno_':[frames[p][paos[p],:] for p in pairs]}
    for name in ['i_Qa_ij_','i_Qk_ij_','K_mibj_','J_ijmb_','K_ivvv_','K_iajb_','L_iajb_','J_ikac_non_proj_','K_iakc_non_proj_','Qma_ij_','Qab_ij_']:d[name]=[]
    for i,j in pairs:
        p=(i,j);x=frames[p];q=fits[p];n=x.shape[1]
        ov=np.einsum('qia,ab->qib',q[:,:no,no:],x)
        vv=np.einsum('ab,qac,cd->qbd',x,q[:,no:,no:],x)
        k=ov[:,i,:].T@ov[:,j,:]
        d['i_Qa_ij_'].append(ov[:,i,:]);d['i_Qk_ij_'].append(q[:,i,:no]);d['K_iajb_'].append(k);d['L_iajb_'].append(2*k-k.T)
        d['K_mibj_'].append(q[:,i,:no].T@ov[:,j,:]);d['J_ijmb_'].append(np.einsum('q,qia->ia',q[:,i,j],ov))
        d['K_ivvv_'].append(np.einsum('qa,qbc->abc',ov[:,i,:],vv).reshape(n,n*n))
        d['Qma_ij_'].append(list(ov));d['Qab_ij_'].append(list(vv))
        js=[];ks=[]
        for k in range(no):
            y=frames[k,j]
            js.append(np.einsum('q,ab,qac,cd->bd',q[:,i,k],x,q[:,no:,no:],y))
            ks.append(np.einsum('qa,qc,cb->ab',ov[:,i,:],q[:,k,no:],y))
        d['J_ikac_non_proj_'].append(js);d['K_iakc_non_proj_'].append(ks)
    t1=[rng.normal(scale=.03,size=(frames[i,i].shape[1],1)) for i in range(no)]
    z1=[rng.normal(scale=.08,size=t.shape) for t in t1];t2=[None]*len(pairs);z2=[None]*len(pairs)
    for p,(i,j) in enumerate(pairs):
        if i>j:continue
        n=frames[i,j].shape[1]
        t=rng.normal(scale=.025,size=(n,n));z=rng.normal(scale=.08,size=(n,n))
        if i==j:t=(t+t.T)/2;z=(z+z.T)/2
        t2[p]=t;t2[rev[p]]=t.T;z2[p]=z;z2[rev[p]]=z.T
    return d,t1,t2,z1,z2

def evaluate(x,kind,threads=1):return native.evaluate(*x,kind,threads)
def maxdiff(a,b):return max(np.max(np.abs(x-y)) for xs,ys in zip(a,b) for x,y in zip(xs,ys))
def zero(xs):return [np.zeros_like(x) for x in xs]
def screened(x,missing=((0,2),(2,0))):
    d,t1,t2,z1,z2=x;d=dict(d);no=len(t1);old=d['ij_to_i_j_'];keep=[p for p,ij in enumerate(old) if ij not in missing]
    pairs=[old[p] for p in keep];idx=np.full((no,no),-1,dtype=int)
    for p,(i,j) in enumerate(pairs):idx[i,j]=p
    domains=[[k for k in range(no) if idx[i,k]>=0 and idx[j,k]>=0] for i,j in pairs]
    dense=[]
    for dom in domains:
        a=[-1]*no
        for k,v in enumerate(dom):a[v]=k
        dense.append(a)
    for name in ['n_pno_','lmopair_to_paos_','lmopair_to_ribfs_','X_pno_','i_Qa_ij_','K_ivvv_','K_iajb_','L_iajb_','Qab_ij_']:
        d[name]=[d[name][p] for p in keep]
    d['i_Qk_ij_']=[d['i_Qk_ij_'][p][:,dom] for p,dom in zip(keep,domains)]
    for name in ['K_mibj_','J_ijmb_']:d[name]=[d[name][p][dom,:] for p,dom in zip(keep,domains)]
    for name in ['J_ikac_non_proj_','K_iakc_non_proj_']:d[name]=[[d[name][p][k] for k in dom] for p,dom in zip(keep,domains)]
    d['Qma_ij_']=[[a[dom,:] for a in d['Qma_ij_'][p]] for p,dom in zip(keep,domains)]
    d.update(ij_to_i_j_=pairs,ij_to_ji_=[int(idx[j,i]) for i,j in pairs],i_j_to_ij_=idx.tolist(),i_j_to_ij_strong_=idx.tolist(),lmopair_to_lmos_=domains,lmopair_to_lmos_dense_=dense)
    return d,t1,[t2[p] for p in keep],z1,[z2[p] for p in keep]
def perturb(x,rank,ds,h):
    y=list(x);y[rank]=[a+h*b for a,b in zip(x[rank],ds)];return y
def dot(a,b):return sum(np.vdot(x,y) for x,y in zip(a[0],b[0]))+.5*sum(np.vdot(x,y) for x,y in zip(a[1],b[1]))
