/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2026 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

// Analytic response of the fixed-branch dipole and crude SC-MP2 estimates.
#include "dlpno.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/molecule.h"
#include <cmath>
#include <numeric>

namespace psi {
namespace dlpno {
namespace screened_detail {
SharedMatrix project(const SharedMatrix& a, const SharedMatrix& x) {
    return linalg::triplet(x, a, x, true, false, false);
}
SharedMatrix lift(const SharedMatrix& a, const SharedMatrix& x) {
    return linalg::triplet(x, a, x, false, false, true);
}
SharedMatrix basis_bar(const SharedMatrix& a, const SharedMatrix& x, const SharedMatrix& bar) {
    auto g = linalg::triplet(a, x, bar, false, false, true);
    g->add(linalg::triplet(a, x, bar, true, false, false));
    return g;
}
SharedMatrix spin(const SharedMatrix& t) {
    auto x = t->clone(); x->scale(2.); x->subtract(t->transpose()); return x;
}
SharedMatrix indices(const std::vector<int>& list) {
    auto x = std::make_shared<Matrix>(list.size(), 1);
    for (int i = 0; i < list.size(); ++i) (*x)(i, 0) = list[i];
    return x;
}
}  // namespace screened_detail

void DLPNOCCSD_T::form_screened_pair_response(const std::vector<double>& energies) {
    const int n = C_pao_->nrow(), no = C_lmo_->ncol(), natom = molecule_->natom();
    const bool keep = options_.get_bool("DLPNO_KEEP_LMP2_RESPONSE");
    ccsd_foo_response_.assign(no, {});
    ccsd_fov_response_.assign(no, {});
    ccsd_fvv_response_.assign(n, {});
    ccsd_overlap_response_.assign(n, {});
    df_pao_coefficient_bar_.assign(n, {});
    fitted_coefficient_bar_.assign(natom, {});
    gradient_lmo_slices_.resize(natom);
    const auto S = reference_wavefunction_->S();
    auto SL = linalg::doublet(S, C_lmo_);
    for (int atom = 0; atom < natom; ++atom) gradient_lmo_slices_[atom] = fitted_lmo_slice(SL, atom);
    screened_pair_blocks_.clear();
    screened_pair_domains_ = {{"AO_OCC", riatom_to_bfs1_}, {"AO_VIR", riatom_to_bfs2_},
                              {"DIPOLE_PAO", dipole_pao_domains_}, {"DIPOLE_PAIRS", {}}, {"SCMP2_PAIRS", {}}};
    MetricShellBlocks metric(ribasis_->nshell());
    std::vector<ThreeCenterDensity> density(ribasis_->nbf());
    double replay = 0., excluded = 0.;
    for (int p = 0; p < ij_to_i_j_.size(); ++p) {
        const auto [i, j] = ij_to_i_j_[p];
        if (i >= j || std::abs(energies[p]) >= T_CUT_PAIRS_MP2_) continue;
        const auto& paos = lmopair_to_paos_[p];
        auto fit = pair_exchange_fit(p);
        auto M = submatrix_rows_and_cols(*S_pao_, paos, paos);
        auto F0 = submatrix_rows_and_cols(*F_pao_, paos, paos);
        auto [X, eps] = orthocanonicalizer(M, F0);
        auto K = screened_detail::project(fit.K, X), t = K->clone();
        const double occ = (*F_lmo_)(i, i) + (*F_lmo_)(j, j);
        for (int a = 0; a < t->nrow(); ++a)
            for (int b = 0; b < t->ncol(); ++b) {
                const double gap = occ - (*eps)(a) - (*eps)(b);
                if (std::abs(gap) <= 1.e-12) throw PSIEXCEPTION("Singular screened SC-MP2 denominator.");
                (*t)(a, b) /= gap;
            }
        const double value = 2. * K->vector_dot(screened_detail::spin(t));  // unique off-diagonal pair
        excluded += value;
        replay = std::max(replay, std::abs(value - 2.*energies[p]));
        auto z = screened_detail::spin(t); z->scale(2.);
        auto kb = z->clone(); kb->scale(2.);
        auto fb = linalg::doublet(z, t, false, true);
        fb->add(linalg::doublet(t, z, true, false)); fb->hermitivitize();
        const double ob = -z->vector_dot(t);
        ccsd_foo_response_[i][i] += ob; ccsd_foo_response_[j][j] += ob;
        add_local_block(ccsd_fvv_response_, paos, paos, screened_detail::lift(fb, X));
        auto xb = screened_detail::basis_bar(fit.K, X, kb); xb->add(screened_detail::basis_bar(F0, X, fb));
        add_local_block(ccsd_overlap_response_, paos, paos, parent_overlap_bar(X, xb));
        auto Kbar = screened_detail::lift(kb, X);
        auto di = linalg::doublet(fit.bj, Kbar, false, true), dj = linalg::doublet(fit.bi, Kbar);
        auto dJ = linalg::doublet(fit.bi, di, false, true); dJ->scale(-1.); dJ->hermitivitize();
        scatter_pair_fit(p, dJ, di, dj, metric, density);
        if (keep) {
            screened_pair_domains_["SCMP2_PAIRS"].push_back({i,j});
            screened_pair_blocks_[{i,j}] = {{"PAOS",screened_detail::indices(paos)}, {"AUX",screened_detail::indices(lmopair_to_ribfs_[p])},
                                            {"PARENT",X}, {"QI",fit.qi}, {"QJ",fit.qj}};
        }
    }
    if (replay > 1.e-10) throw PSIEXCEPTION("Screened SC-MP2 energy replay disagrees with the forward energy.");
    set_scalar_variable("DLPNO SCREENED SC-MP2 REPLAY ERROR", replay);
    set_scalar_variable("DLPNO SCREENED SC-MP2 RESPONSE ENERGY", excluded);
    for (const auto& kind : {"AUXILIARY METRIC", "THREE-CENTER"}) {
        const std::string name = std::string("DLPNO SC-LMP2 PAIRS ") + kind + " GRADIENT";
        set_array_variable(name, std::string(kind) == "AUXILIARY METRIC" ? contract_metric(metric, name)
                                                                       : contract_three_center(density, name));
    }
    finish_df_refit_response();  // still uses the CRUDE maps and coefficient refits

    MintsHelper mints(basisset_, options_);
    auto dip = mints.ao_dipole();
    std::vector<SharedMatrix> db(3);
    for (auto& b : db) b = std::make_shared<Matrix>(n,n);
    screened_lmo_bar_ = std::make_shared<Matrix>(n,no);
    std::vector<SharedMatrix> trans(no), frame(no);
    auto centers = std::make_shared<Matrix>(no,3);
    for (int i = 0; i < no; ++i) {
        const auto& paos = dipole_pao_domains_[i];
        const auto& X = dipole_pao_parent_[i];
        frame[i] = linalg::doublet(submatrix_cols(*C_pao_, paos), X);
        auto l = submatrix_cols(*C_lmo_, {i});
        trans[i] = std::make_shared<Matrix>(3, X->ncol());
        for (int k = 0; k < 3; ++k) {
            (*centers)(i,k) = l->vector_dot(linalg::doublet(dip[k], l));
            auto row = linalg::triplet(l, dip[k], frame[i], true, false, false);
            for (int a = 0; a < X->ncol(); ++a) (*trans[i])(k,a) = (*row)(0,a);
        }
        if (keep) screened_pair_blocks_[{i,i}] = {{"PARENT",X}};
    }
    double distant = 0.;
    for (int i = 0; i < no; ++i) for (int j = i+1; j < no; ++j) {
        if (i_j_to_ij_[i][j] >= 0) continue;
        if (keep) screened_pair_domains_["DIPOLE_PAIRS"].push_back({i,j});
        double R[3], r2 = 0.;
        for (int k = 0; k < 3; ++k) { R[k] = (*centers)(i,k)-(*centers)(j,k); r2 += R[k]*R[k]; }
        if (r2 <= 1.e-20) throw PSIEXCEPTION("Coincident centroids in an excluded dipole pair.");
        const double r3 = std::pow(r2,-1.5), r5 = std::pow(r2,-2.5), r7 = std::pow(r2,-3.5);
        auto A = std::make_shared<Matrix>(3,3);
        for (int k=0;k<3;++k) for (int l=0;l<3;++l) (*A)(k,l) = (k==l ? r3 : 0.)-3.*r5*R[k]*R[l];
        auto K = linalg::triplet(trans[i], A, trans[j], true, false, false), t = K->clone();
        auto Fi = screened_detail::project(F_ao_, frame[i]), Fj = screened_detail::project(F_ao_, frame[j]);
        const double occ = (*F_lmo_)(i,i)+(*F_lmo_)(j,j);
        for (int a=0;a<t->nrow();++a) for (int b=0;b<t->ncol();++b) {
            const double gap=occ-(*Fi)(a,a)-(*Fj)(b,b);
            if (std::abs(gap)<=1.e-12) throw PSIEXCEPTION("Singular dipole pair denominator.");
            (*t)(a,b)/=gap;
        }
        distant += 4.*K->vector_dot(t);
        auto z=t->clone(); z->scale(4.);
        auto kb=t->clone(); kb->scale(8.);
        auto fbi=linalg::doublet(z,t,false,true), fbj=linalg::doublet(t,z,true,false);
        fbi->hermitivitize(); fbj->hermitivitize();
        double ob=-z->vector_dot(t);
        ccsd_foo_response_[i][i]+=ob; ccsd_foo_response_[j][j]+=ob;
        auto ti=linalg::triplet(A,trans[j],kb,false,false,true);
        auto tj=linalg::triplet(A,trans[i],kb,true,false,false);
        auto ab=linalg::triplet(trans[i],kb,trans[j],false,false,true);
        double trace=0., quadratic=0., rb[3]{};
        for(int k=0;k<3;++k) { trace+=(*ab)(k,k); for(int l=0;l<3;++l) quadratic+=R[k]*(*ab)(k,l)*R[l]; }
        for(int k=0;k<3;++k) {
            rb[k]=-3.*r5*trace*R[k]+15.*r7*quadratic*R[k];
            for(int l=0;l<3;++l) rb[k]-=3.*r5*((*ab)(k,l)+(*ab)(l,k))*R[l];
        }
        for (int side=0;side<2;++side) {
            const int m=side?j:i; const double sign=side?-1.:1.;
            auto tb=side?tj:ti, fb=side?fbj:fbi;
            const auto& paos=dipole_pao_domains_[m]; const auto& X=dipole_pao_parent_[m];
            auto N=submatrix_cols(*C_pao_,paos), L=submatrix_cols(*C_lmo_,{m});
            auto rawbar=linalg::doublet(tb,X,false,true);
            auto F0=submatrix_rows_and_cols(*F_pao_,paos,paos);
            auto xb=screened_detail::basis_bar(F0,X,fb);
            add_local_block(ccsd_fvv_response_,paos,paos,screened_detail::lift(fb,X));
            for(int k=0;k<3;++k) {
                auto b=submatrix_rows(*rawbar,{k})->transpose();
                auto y=linalg::doublet(N,b), dl=linalg::doublet(dip[k],L);
                auto lb=linalg::doublet(dip[k],y); lb->axpy(2.*sign*rb[k],dl);
                for(int a=0;a<n;++a) (*screened_lmo_bar_)(a,m)+=(*lb)(a,0);
                auto nb=linalg::doublet(dl,b,false,true);
                std::vector<int> rows(n); std::iota(rows.begin(),rows.end(),0);
                add_local_block(df_pao_coefficient_bar_,rows,paos,nb);
                db[k]->add(linalg::doublet(L,y,false,true));
                db[k]->axpy(sign*rb[k],linalg::doublet(L,L,false,true));
                auto raw=linalg::triplet(L,dip[k],N,true,false,false);
                xb->add(linalg::doublet(raw,submatrix_rows(*tb,{k}),true,false));
            }
            add_local_block(ccsd_overlap_response_,paos,paos,parent_overlap_bar(X,xb));
        }
    }
    replay=std::abs(distant-de_dipole_);
    if(replay>1.e-10) throw PSIEXCEPTION("Unique dipole-pair replay disagrees with the forward energy.");
    set_scalar_variable("DLPNO DIPOLE PAIR REPLAY ERROR",replay);
    set_scalar_variable("DLPNO DIPOLE PAIR RESPONSE ENERGY",distant);
    auto explicit_gradient=std::make_shared<Matrix>(natom,3);
    for(int atom=0;atom<natom;++atom) {
        auto derivative=mints.ao_elec_dip_deriv1(atom);
        for(int k=0;k<3;++k) for(int xyz=0;xyz<3;++xyz)
            (*explicit_gradient)(atom,xyz)+=db[k]->vector_dot(derivative[3*k+xyz]);
    }
    set_array_variable("DLPNO DIPOLE PAIRS EXPLICIT GRADIENT",explicit_gradient);
    screened_pair_response_={std::move(ccsd_foo_response_),std::move(ccsd_fov_response_),
        std::move(ccsd_fvv_response_),std::move(ccsd_overlap_response_),std::move(df_pao_coefficient_bar_),
        std::move(df_sclmo_bar_),std::move(df_refit_overlap_bar_)};
    if (!keep) screened_pair_domains_.clear();
    dipole_pao_domains_.clear();
    dipole_pao_parent_.clear();
}

void DLPNOCCSD_T::merge_screened_pair_response() {
    auto merge=[](SparseResponseRows& dst,const SparseResponseRows& src) {
        if(dst.size()!=src.size()) throw PSIEXCEPTION("Inconsistent screened-pair response dimensions.");
        for(int i=0;i<src.size();++i) for(const auto& [j,g]:src[i]) dst[i][j]+=g;
    };
    const auto& r=screened_pair_response_;
    merge(ccsd_foo_response_,r.foo); merge(ccsd_fov_response_,r.fov); merge(ccsd_fvv_response_,r.fvv);
    merge(ccsd_overlap_response_,r.overlap); merge(df_pao_coefficient_bar_,r.coefficients);
    merge(df_sclmo_bar_,r.sclmo); merge(df_refit_overlap_bar_,r.refit_overlap);
    screened_pair_response_ = {};
}

std::tuple<std::map<std::string,SparseMap>,DLPNOCCSD_T::ScreenedPairBlocks>
DLPNOCCSD_T::screened_pair_state() const {
    if(!gradient_model_ || screened_pair_domains_.empty())
        throw PSIEXCEPTION("Screened-pair state requires an RHF gradient model with DLPNO_KEEP_LMP2_RESPONSE.");
    auto copies=screened_pair_blocks_;
    for(auto& [pair,blocks]:copies) for(auto& [label,matrix]:blocks) matrix=matrix->clone();
    return {screened_pair_domains_,copies};
}
} // namespace dlpno
} // namespace psi
