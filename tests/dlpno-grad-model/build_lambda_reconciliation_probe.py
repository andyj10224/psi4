"""Compile verbatim production Lambda/right-residual functions for algebra audits.

Small storage/API adapters only; this is not a molecular Psi4 build or benchmark.
"""
from pathlib import Path
import re, subprocess, sysconfig

import argparse, tempfile
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build-dir',type=Path)
parser.add_argument('--pybind-include',type=Path)
parser.add_argument('--ref',help='Optional git commit to extract for a published-mode regression')
args=parser.parse_args()
root=Path(__file__).resolve().parents[2]
work=args.build_dir or Path(tempfile.mkdtemp(prefix='dlpno-lambda-'))
work.mkdir(parents=True,exist_ok=True)
src=root/'psi4/src/psi4/dlpno'
def source(name):
    if args.ref:
        return subprocess.check_output(['git','show',f'{args.ref}:psi4/src/psi4/dlpno/{name}'],cwd=root,text=True)
    return (src/name).read_text()
right=source('ccsd.cc');left=source('ccsd_lambda.cc')
def function(s,name):
    m=re.search(r'^[^\n]* DLPNOCCSD::'+name+r'\(',s,re.M)
    a=m.start();b=s.index('\n}',a)+2
    return s[a:b]
right_names=['S_PNO','compute_pno_overlaps','t1_ints','t1_fock','compute_beta','compute_gamma','compute_delta','compute_Fkj_double_tilde','compute_R_ia','compute_R_iajb','prepare_ccsd_residual','singles_fock_energy']
left_names=['form_goo','compute_lambda_intermediates','compute_L_ia','compute_alpha_ijkl','compute_L_iajb','gradient_lambda_residual','reverse_R1','reverse_R2','reverse_pair_intermediates','reverse_t1_transform','ccsd_lagrangian','prepare_gradient_lambda_overlaps']
functions=[function(right,n) for n in right_names]+[function(left,n) for n in left_names]
decls='\n'.join(f[:f.index('{')].replace('DLPNOCCSD::','')+';' for f in functions)
decls=decls.replace('bool include_source,','bool include_source = true,').replace('CCAdjoints* workspace)', 'CCAdjoints* workspace = nullptr)')
decls=decls.replace('std::vector<SharedMatrix>* right1,','std::vector<SharedMatrix>* right1 = nullptr,').replace('std::vector<SharedMatrix>* right2)', 'std::vector<SharedMatrix>* right2 = nullptr)')
decls=decls.replace('bool exact_residual','bool exact_residual = false').replace('bool include_source)', 'bool include_source = true)')
header=r'''
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <map>
#include <set>
#include <mutex>
#include <cstring>
#include <omp.h>
#define PSIEXCEPTION(x) std::runtime_error(x)
namespace psi {
struct Environment {int get_n_threads() const {return omp_get_max_threads();}};
struct Process {static inline Environment environment;};
struct Output {template<class...T>void Printf(const char*,T...) {}};
inline Output output; inline Output* outfile=&output;
inline void timer_on(const char*){} inline void timer_off(const char*){}
inline void C_DAXPY(int n,double a,double*x,int ix,double*y,int iy){for(int i=0;i<n;i++)y[i*iy]+=a*x[i*ix];}
inline void C_DGER(int m,int n,double a,double*x,int ix,double*y,int iy,double*z,int ld){for(int i=0;i<m;i++)for(int j=0;j<n;j++)z[i*ld+j]+=a*x[i*ix]*y[j*iy];}
inline void C_DGEMV(char t,int m,int n,double a,double*x,int ld,double*v,int iv,double b,double*y,int iy){int rows=t=='N'?m:n,cols=t=='N'?n:m;for(int i=0;i<rows;i++){double s=0;for(int j=0;j<cols;j++)s+=(t=='N'?x[i*ld+j]:x[j*ld+i])*v[j*iv];y[i*iy]=a*s+b*y[i*iy];}}
class Matrix;using SharedMatrix=std::shared_ptr<Matrix>;
class Matrix {
 int r,c;std::vector<double>d;
public:
 Matrix(int a,int b):r(a),c(b),d(size_t(a)*b,0.){if(a<0||b<0)throw PSIEXCEPTION("Bad shape");}
 Matrix(const char*,int a,int b):Matrix(a,b){}
 int nrow() const{return r;} int ncol() const{return c;} int nirrep() const{return 1;}
 double& operator()(int a,int b){if(a<0||a>=r||b<0||b>=c)throw PSIEXCEPTION("Bad index");return d[size_t(a)*c+b];}
 double get(int a,int b)const{if(a<0||a>=r||b<0||b>=c)throw PSIEXCEPTION("Bad index");return d[size_t(a)*c+b];}
 void set(int a,int b,double x){(*this)(a,b)=x;}
 double* get_pointer(){return d.data();}
 void zero(){std::fill(d.begin(),d.end(),0.);}
 void scale(double a){for(auto&x:d)x*=a;}
 SharedMatrix clone()const{return std::make_shared<Matrix>(*this);}
 SharedMatrix transpose()const{auto o=std::make_shared<Matrix>(c,r);for(int i=0;i<r;i++)for(int j=0;j<c;j++)(*o)(j,i)=get(i,j);return o;}
 void transpose_this(){*this=*transpose();}
 void reshape(int a,int b){if(size_t(a)*b!=d.size())throw PSIEXCEPTION("Bad reshape");r=a;c=b;}
 void copy(const SharedMatrix&b){*this=*b;}
 void axpy(double a,const SharedMatrix&b){if(r!=b->r||c!=b->c)throw PSIEXCEPTION("Bad axpy");for(size_t i=0;i<d.size();i++)d[i]+=a*b->d[i];}
 void add(const SharedMatrix&b){axpy(1.,b);} void subtract(const SharedMatrix&b){axpy(-1.,b);}
 double vector_dot(const SharedMatrix&b)const{if(r!=b->r||c!=b->c)throw PSIEXCEPTION("Bad dot");double v=0;for(size_t i=0;i<d.size();i++)v+=d[i]*b->d[i];return v;}
 void set_diagonal(const std::vector<double>&e){for(int i=0;i<r;i++)(*this)(i,i)=e[i];}
 void gemm(bool ta,bool tb,double alpha,const SharedMatrix&a,const SharedMatrix&b,double beta);
};
namespace linalg {
inline SharedMatrix doublet(const SharedMatrix&a,const SharedMatrix&b,bool ta=false,bool tb=false){
 if(!a||!b)throw PSIEXCEPTION("Null doublet");int r=ta?a->ncol():a->nrow(),k=ta?a->nrow():a->ncol(),c=tb?b->nrow():b->ncol();
 if(k!=(tb?b->ncol():b->nrow()))throw PSIEXCEPTION("Bad doublet");auto o=std::make_shared<Matrix>(r,c);
 for(int i=0;i<r;i++)for(int j=0;j<c;j++)for(int l=0;l<k;l++)(*o)(i,j)+=(ta?a->get(l,i):a->get(i,l))*(tb?b->get(j,l):b->get(l,j));return o;
}
inline SharedMatrix triplet(const SharedMatrix&a,const SharedMatrix&b,const SharedMatrix&c,bool ta=false,bool tb=false,bool tc=false){return doublet(doublet(a,b,ta,tb),c,false,tc);}
}
inline void Matrix::gemm(bool ta,bool tb,double alpha,const SharedMatrix&a,const SharedMatrix&b,double beta){auto x=linalg::doublet(a,b,ta,tb);scale(beta);axpy(alpha,x);}
inline SharedMatrix submatrix_rows_and_cols(const Matrix&a,const std::vector<int>&r,const std::vector<int>&c){auto o=std::make_shared<Matrix>(r.size(),c.size());for(int i=0;i<r.size();i++)for(int j=0;j<c.size();j++)(*o)(i,j)=a.get(r[i],c[j]);return o;}
inline std::vector<int> range(int n){std::vector<int>r(n);std::iota(r.begin(),r.end(),0);return r;}
inline SharedMatrix submatrix_rows(const Matrix&a,const std::vector<int>&r){return submatrix_rows_and_cols(a,r,range(a.ncol()));}
inline SharedMatrix submatrix_cols(const Matrix&a,const std::vector<int>&c){return submatrix_rows_and_cols(a,range(a.nrow()),c);}
inline std::vector<int>merge_lists(std::vector<int>a,const std::vector<int>&b){a.insert(a.end(),b.begin(),b.end());std::sort(a.begin(),a.end());a.erase(std::unique(a.begin(),a.end()),a.end());return a;}
inline std::vector<int>index_list(const std::vector<int>&a,const std::vector<int>&b){std::vector<int>v;for(int x:b)v.push_back(std::find(a.begin(),a.end(),x)-a.begin());return v;}
namespace dlpno {
using Blocks=std::vector<SharedMatrix>;
class DLPNOCCSD {
public:
 struct CCAdjoints;
 struct GradientLambdaFixed {
  std::vector<Blocks>q;Blocks f;std::mutex overlap_mutex;
  std::map<std::pair<int,int>,SharedMatrix>overlaps;
  std::set<std::pair<int,int>>overlap_requests;
  struct OverlapPanel {SharedMatrix left;std::map<int,int>columns;};
  std::map<int,OverlapPanel>overlap_panels;
  size_t limit=0,bytes=0,overlap_hits=0,overlap_builds=0,overlap_panel_uses=0;
 };
 std::unique_ptr<GradientLambdaFixed>gradient_lambda_fixed_;
 int nalpha_;int nfrzc()const{return 0;}
 bool lambda_requested_=true,gradient_model_=true,low_memory_overlap_=false;
 std::vector<std::pair<int,int>>ij_to_i_j_;
 std::vector<int>ij_to_ji_,n_pno_;
 std::vector<std::vector<int>>i_j_to_ij_,i_j_to_ij_strong_,lmopair_to_lmos_,lmopair_to_lmos_dense_,lmopair_to_paos_,lmopair_to_ribfs_;
 std::vector<std::vector<double>>e_pno_;
 SharedMatrix F_im_double_tilde_,F_lmo_,F_lmo_pao_,F_pao_,Fkj_,S_pao_,rho_oo_;
 Blocks F_vv_double_tilde_,Fab_,Fai_,Fkc_,Fkc_bar_,J_ijmb_,J_kaec_bar_,J_kmic_bar_,K_eimn_dt_,K_iajb_,K_ijmb_bar_,K_ivvv_,K_maef_dt_,K_mibj_,L_iajb_,L_ieab_bar_,M_kace_bar_,M_mkic_bar_,T_ia_,T_iajb_,T_n_ij_,Tt_iajb_,X_pno_,beta_,delta_,delta_imae_tilde_,gamma_,i_Qa_ij_,i_Qa_t1_,i_Qk_ij_,i_Qk_t1_,lambda_ia_,lambda_iajb_,lambda_iajb_bar_,rho_vv_,triples_source1_,triples_source2_;
 std::vector<Blocks>F_knia_hat_,J_ikac_non_proj_,K_iakc_non_proj_,delta_double_tilde_,gamma_double_tilde_,Qma_ij_,Qab_ij_;
 Blocks lambda_gamma_,lambda_delta_;
 std::vector<Blocks>S_pno_ij_kj_,S_pno_ij_nn_,S_pno_ij_mn_;
 Blocks QIA_PNO(int p){return Qma_ij_[std::min(p,ij_to_ji_[p])];}
 Blocks QAB_PNO(int p){return Qab_ij_[std::min(p,ij_to_ji_[p])];}
 SharedMatrix gradient_lambda_matrix(int,int,int,const std::function<SharedMatrix()>&f){return f();}
'''
header+=decls+'\n};\n}\n}\n'
adj=source('ccsd_reverse.h');adj=adj[adj.index('namespace psi {'):]
module=r'''
namespace py=pybind11;
using Array=py::array_t<double,py::array::c_style|py::array::forcecast>;
using namespace psi;using namespace psi::dlpno;
SharedMatrix mat(py::handle h){auto a=py::cast<Array>(h);if(a.ndim()!=2)throw PSIEXCEPTION("Expected matrix");auto m=std::make_shared<Matrix>(a.shape(0),a.shape(1));std::copy_n(a.data(),a.size(),m->get_pointer());return m;}
Blocks blocks(py::handle h){Blocks b;for(auto v:py::cast<py::list>(h))b.push_back(mat(v));return b;}
std::vector<Blocks> nested(py::handle h){std::vector<Blocks>b;for(auto v:py::cast<py::list>(h))b.push_back(blocks(v));return b;}
py::list arrays(const Blocks&b){py::list out;for(auto&m:b){Array a({m->nrow(),m->ncol()});std::copy_n(m->get_pointer(),a.size(),a.mutable_data());out.append(a);}return out;}
PYBIND11_MODULE(lambda_probe,m){
m.def("evaluate",[](py::dict data,py::list t1,py::list t2,py::list z1,py::list z2,std::string kind,int threads){
 omp_set_num_threads(threads);
 DLPNOCCSD w;
'''
for n in ['ij_to_i_j_']:module+=f'w.{n}=py::cast<std::vector<std::pair<int,int>>>(data["{n}"]);\n'
for n in ['ij_to_ji_','n_pno_']:module+=f'w.{n}=py::cast<std::vector<int>>(data["{n}"]);\n'
for n in ['i_j_to_ij_','i_j_to_ij_strong_','lmopair_to_lmos_','lmopair_to_lmos_dense_','lmopair_to_paos_','lmopair_to_ribfs_']:module+=f'w.{n}=py::cast<std::vector<std::vector<int>>>(data["{n}"]);\n'
for n in ['F_lmo_','F_lmo_pao_','F_pao_','S_pao_']:module+=f'w.{n}=mat(data["{n}"]);\n'
for n in ['X_pno_','i_Qa_ij_','i_Qk_ij_','K_mibj_','J_ijmb_','K_ivvv_','K_iajb_','L_iajb_']:module+=f'w.{n}=blocks(data["{n}"]);\n'
for n in ['J_ikac_non_proj_','K_iakc_non_proj_','Qma_ij_','Qab_ij_']:module+=f'w.{n}=nested(data["{n}"]);\n'
module+=r'''
 w.low_memory_overlap_=data.contains("low_memory_overlap") && py::cast<bool>(data["low_memory_overlap"]);
 w.compute_pno_overlaps();
 w.T_ia_=blocks(t1);w.T_iajb_=blocks(t2);w.lambda_ia_=blocks(z1);w.lambda_iajb_=blocks(z2);
 w.nalpha_=t1.size();int np=t2.size();w.i_Qa_t1_.resize(np);w.i_Qk_t1_.resize(np);w.T_n_ij_.resize(np);w.e_pno_.resize(np);
 for(auto&t:w.T_iajb_){auto u=t->clone();u->scale(2.);u->subtract(t->transpose());w.Tt_iajb_.push_back(u);}
 for(auto&z:w.lambda_iajb_){auto b=z->transpose();b->axpy(.5,z);w.lambda_iajb_bar_.push_back(b);}
 using namespace psi::dlpno::ccsd_detail;
 auto a=zero_blocks(w.T_ia_),b=zero_blocks(w.T_iajb_),bn=zero_blocks(w.T_iajb_);
 std::vector<Blocks>buf(threads);for(auto&v:buf)v=zero_blocks(w.T_ia_);
 if(kind=="lagrangian")return py::object(py::float_(w.ccsd_lagrangian(w.lambda_ia_,w.lambda_iajb_)));
 if(kind=="right"){w.prepare_ccsd_residual();w.compute_R_ia(a,buf);w.compute_R_iajb(b,bn);}
 else if(kind=="published"||kind=="corrected"||kind=="corrected_split"){
   bool exact=kind!="published";
   w.compute_lambda_intermediates(exact);
   if(exact && data.contains("cache_bytes")) {
     w.gradient_lambda_fixed_=std::make_unique<DLPNOCCSD::GradientLambdaFixed>();
     auto& fixed=*w.gradient_lambda_fixed_;fixed.limit=py::cast<size_t>(data["cache_bytes"]);fixed.q.resize(np);fixed.f.resize(np);
     for(int p=0;p<np;p++) {
       auto [i,j]=w.ij_to_i_j_[p];if(i>j)continue;
       fixed.f[p]=w.F_vv_double_tilde_[p];
       size_t bytes=w.QAB_PNO(p).size()*w.n_pno_[p]*w.n_pno_[p]*sizeof(double);
       if(bytes>fixed.limit-fixed.bytes)continue;
       auto ov=w.QIA_PNO(p),vv=w.QAB_PNO(p);
       for(int q=0;q<vv.size();q++){auto v=vv[q]->clone();v->subtract(linalg::doublet(w.T_n_ij_[p],ov[q],true,false));fixed.q[p].push_back(v);}
       fixed.bytes+=bytes;
     }
   }
   for(int replay=0;replay<(exact?3:1);++replay) {
     w.form_goo(exact);w.compute_L_ia(a,buf,exact);w.compute_L_iajb(b,bn,exact);
     if(exact && w.gradient_lambda_fixed_ && replay==0)w.prepare_gradient_lambda_overlaps();
   }
   if(kind=="corrected_split") {
     auto z1=w.lambda_ia_,z2=w.lambda_iajb_;
     w.lambda_ia_=zero_blocks(z1);w.lambda_iajb_=zero_blocks(z2);
     auto source1=zero_blocks(a),source2=zero_blocks(b);
     w.form_goo(true);w.compute_L_ia(source1,buf,true);w.compute_L_iajb(source2,bn,true);
     w.lambda_ia_=z1;w.lambda_iajb_=z2;
     w.form_goo(true);w.compute_L_ia(a,buf,true,false);w.compute_L_iajb(b,bn,true,false);
     for(int i=0;i<a.size();i++)a[i]->add(source1[i]);
     for(int p=0;p<b.size();p++)b[p]->add(source2[p]);
   }
 }else{
   w.prepare_ccsd_residual();w.beta_=w.compute_beta();w.gamma_=w.compute_gamma();w.delta_=w.compute_delta();w.F_im_double_tilde_=w.compute_Fkj_double_tilde();
   w.triples_source1_=zero_blocks(a);w.triples_source2_=zero_blocks(b);
   w.gradient_lambda_residual(w.lambda_ia_,w.lambda_iajb_,a,b,true);
 }
 return py::object(py::make_tuple(arrays(a),arrays(b)));
},py::arg("data"),py::arg("t1"),py::arg("t2"),py::arg("z1"),py::arg("z2"),py::arg("kind"),py::arg("threads")=1);
}
'''
if 'form_goo(bool exact_residual)' not in left:
    # Adapt only the test wrapper to the historical public signatures. The
    # extracted production equations remain verbatim; corrected mode is absent.
    module=module.replace('bool exact=kind!="published";',
                          'bool exact=false; if(kind!="published")throw PSIEXCEPTION("Historical source has no corrected mode");')
    module=module.replace('w.compute_lambda_intermediates(exact);','w.compute_lambda_intermediates();')
    module=re.sub(r'w.form_goo\((exact|true)\)', 'w.form_goo()',module)
    module=re.sub(r'w.compute_L_ia\((a|source1),buf,(exact|true)(,false)?\)',r'w.compute_L_ia(\1,buf)',module)
    module=re.sub(r'w.compute_L_iajb\((b|source2),bn,(exact|true)(,false)?\)',r'w.compute_L_iajb(\1,bn)',module)
text=header+adj+'\nnamespace psi { namespace dlpno {\nusing namespace ccsd_detail;\n'+'\n'.join(functions)+'\n}}\n'+module
(work/'probe.cc').write_text(text)
if args.pybind_include is None:
    import pybind11
    inc=Path(pybind11.get_include())
else:inc=args.pybind_include
cmd=['g++','-O1','-g','-std=c++17','-shared','-fPIC','-fopenmp','-I'+str(inc),'-I'+sysconfig.get_paths()['include'],str(work/'probe.cc'),'-o',str(work/('lambda_probe'+sysconfig.get_config_var('EXT_SUFFIX')))]
subprocess.run(cmd,check=True)
print(work)
