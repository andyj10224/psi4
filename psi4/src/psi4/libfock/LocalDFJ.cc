#include "composite.h"

#include "psi4/libmints/integral.h"
#include "psi4/libmints/vector.h"
#include "psi4/lib3index/dftensor.h"
#include "psi4/libqt/qt.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include <vector>
#include <unordered_set>
#include <algorithm>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#include "psi4/libpsi4util/process.h"
#endif

using namespace psi;

namespace psi {

LocalDFJ::LocalDFJ(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary, 
               Options& options) : DirectDFJ(primary, auxiliary, options) {

    df_cfmm_tree_ = std::make_shared<CFMMTree>(primary_, auxiliary_, options_);
    molecule_ = primary_->molecule();
    r0_ = options_.get_double("LOCAL_DF_R0");
    r1_ = options_.get_double("LOCAL_DF_R1");

    setup_local_regions();
    build_atomic_inverse();
    rho_a_L_.resize(molecule_->natom());
    I_KX_.resize(molecule_->natom());
}

double bump_function(double x, double r0, double r1) {
    double bump_value;
    if (x <= r0) {
        bump_value = 1.0;
    } else if (x >= r1) {
        bump_value = 0.0;
    } else {
        double temp1 = (r1 - r0) / (r1 - x);
        double temp2 = (r1 - r0) / (x - r0);
        bump_value = 1.0 / (1.0 + std::exp(temp1 - temp2));
    }
    return bump_value;
}

void LocalDFJ::setup_local_regions() {

    unsigned natom = molecule_->natom();
    unsigned pri_nshell = primary_->nshell();
    unsigned aux_nshell = auxiliary_->nshell();

    auto aux_shell_pairs = ints_[0]->shell_pairs_bra();
    auto pri_shell_pairs = ints_[0]->shell_pairs_ket();

    atom_to_aux_shells_.resize(natom);
    aux_shell_to_atoms_.resize(aux_nshell);
    atom_aux_shell_bump_value_.resize(natom);
    atom_aux_shell_function_offset_.resize(natom);
    for (int atom = 0; atom < natom; atom++) {
        Vector3 Ratom = molecule_->xyz(atom);
        int offset = 0;
        for (const auto aux_pair : aux_shell_pairs) {
            int Q = aux_pair.first;
            Vector3 R_Q = auxiliary_->shell(Q).center();
            Vector3 dR = Ratom - R_Q;
            double dist2 = dR.dot(dR);
            if (dist2 <= r1_ * r1_) {
                atom_to_aux_shells_[atom].push_back(Q);
                aux_shell_to_atoms_[Q].push_back(atom);
                atom_aux_shell_function_offset_[atom][Q] = offset;
                atom_aux_shell_bump_value_[atom][Q] = bump_function(std::sqrt(dist2), r0_, r1_);
                offset += auxiliary_->shell(Q).nfunction();
            }
        }
    }

    naux_per_atom_.resize(natom);
    for (int atom = 0; atom < natom; atom++) {
        int naux = 0;
        for (const auto& L_a : atom_to_aux_shells_[atom]) {
            naux += auxiliary_->shell(L_a).nfunction();
        }
        naux_per_atom_[atom] = naux;
    }

    atom_to_pri_shells_.resize(natom);
    for (const auto& pri_pair : pri_shell_pairs) {
        int U = pri_pair.first;
        int V = pri_pair.second;

        int Ucenter = primary_->shell(U).ncenter();
        int Vcenter = primary_->shell(V).ncenter();

        atom_to_pri_shells_[Ucenter].push_back(U * pri_nshell + V);      
    }
}

void LocalDFJ::build_atomic_inverse() {
    unsigned natom = molecule_->natom();
    unsigned naux = auxiliary_->nbf();
    unsigned aux_nshell = auxiliary_->nshell();

    J_X_.resize(natom);
    for (int atom = 0; atom < natom; atom++) {
        int atomic_naux = naux_per_atom_[atom];
        J_X_[atom] = std::make_shared<Matrix>(atomic_naux, atomic_naux);
        J_X_[atom]->zero();
    }

    double** Jmetp = Jmet_->pointer();

#pragma omp parallel for
    for (int atom = 0; atom < natom; atom++) {
        double** JXp = J_X_[atom]->pointer();

        for (int K : atom_to_aux_shells_[atom]) {
            int k_start = auxiliary_->shell(K).start();
            int num_k = auxiliary_->shell(K).nfunction();
            int k_off = atom_aux_shell_function_offset_[atom][K];
            double bump_k = atom_aux_shell_bump_value_[atom][K];
            int Katom = auxiliary_->shell(K).ncenter();

            for (int L : atom_to_aux_shells_[atom]) {
                int l_start = auxiliary_->shell(L).start();
                int num_l = auxiliary_->shell(L).nfunction();
                int l_off = atom_aux_shell_function_offset_[atom][L];
                double bump_l = atom_aux_shell_bump_value_[atom][L];
                int Latom = auxiliary_->shell(L).ncenter();

                double prefactor = (Katom == Latom) ? 1.0 : bump_k * bump_l;

                for (int dk = 0; dk < num_k; dk++) {
                    for (int dl = 0; dl < num_l; dl++) {
#pragma omp atomic
                        JXp[k_off + dk][l_off + dl] += prefactor * Jmetp[k_start + dk][l_start + dl];
                    }
                }
            }
        }
    }
}

void LocalDFJ::build_rho_a_L(const std::vector<SharedMatrix>& D) {
    unsigned natom = molecule_->natom();
    unsigned pri_nshell = primary_->nshell();
    unsigned aux_nshell = auxiliary_->nshell();

    for (int atom = 0; atom < natom; atom++) {
        bool uninitialized = (rho_a_L_[atom].size() == 0);
        for (int i = 0; i < D.size(); i++) {
            if (uninitialized) rho_a_L_[atom].push_back(std::make_shared<Matrix>(naux_per_atom_[atom], 1));
            rho_a_L_[atom][i]->zero();
        }
    }

    // maximum values of Density matrix for shell pair block UV
    // TODO: Integrate this more smoothly into current density screening framework
    Matrix D_max(pri_nshell, pri_nshell);
    auto D_maxp = D_max.pointer();

    if (density_screening_) {
#pragma omp parallel for
        for (size_t UV = 0; UV < pri_nshell * pri_nshell; UV++) {
            size_t U = UV / pri_nshell;
            size_t V = UV % pri_nshell;

            int u_start = primary_->shell_to_basis_function(U);
            int num_u = primary_->shell(U).nfunction();
	        
            int v_start = primary_->shell_to_basis_function(V);
            int num_v = primary_->shell(V).nfunction();

	        for (size_t i = 0; i < D.size(); i++) {
                auto Dp = D[i]->pointer();
                for (size_t u = u_start; u < u_start + num_u; u++) {
                    for (size_t v = v_start; v < v_start + num_v; v++) {
                        D_maxp[U][V] = std::max(D_maxp[U][V], std::abs(Dp[u][v]));
                    }
                }
            }

        }
    }

#pragma omp parallel for
    for (int L = 0; L < aux_nshell; L++) {
        int l_start = auxiliary_->shell(L).start();
        int num_l = auxiliary_->shell(L).nfunction();

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        for (const int& atom : aux_shell_to_atoms_[L]) {
            int l_off = atom_aux_shell_function_offset_[atom][L];

            for (const int& UV_a : atom_to_pri_shells_[atom]) {
                int U = UV_a / pri_nshell;
                int V = UV_a % pri_nshell;

                if (density_screening_) {
    		        double screen_val = D_maxp[U][V] * D_maxp[U][V] * Jmet_max_[L] * ints_[thread]->shell_pair_value(U,V);
                    if (screen_val < cutoff_ * cutoff_) continue;
                }

                double prefactor = (U == V) ? 1.0 : 2.0;

                int u_start = primary_->shell(U).start();
                int num_u = primary_->shell(U).nfunction();

                int v_start = primary_->shell(V).start();
                int num_v = primary_->shell(V).nfunction();

                ints_[thread]->compute_shell(L, 0, U, V);
                const double* buffer = ints_[thread]->buffer();

                for (int i = 0; i < D.size(); i++) {
                    double** Dp = D[i]->pointer();
                    double* raLp = rho_a_L_[atom][i]->pointer()[0];
                    double* Luv = const_cast<double *>(buffer);

                    for (int dl = 0; dl < num_l; dl++) {
                        for (int du = 0; du < num_u; du++) {
                            for (int dv = 0; dv < num_v; dv++) {
#pragma omp atomic
                                raLp[l_off + dl] += prefactor * (*Luv) * Dp[u_start + du][v_start + dv];
                                (Luv)++;
                            }
                        }
                    }
                }
            }
        }
    }
}

void LocalDFJ::build_rho_tilde_K() {
    unsigned natom = molecule_->natom();
    unsigned nmat = rho_a_L_[0].size();
    unsigned naux = auxiliary_->nbf();

    bool uninitialized = (rho_tilde_K_.size() == 0);
    for (int i = 0; i < nmat; i++) {
        if (uninitialized) rho_tilde_K_.push_back(std::make_shared<Matrix>(naux, 1));
        rho_tilde_K_[i]->zero();
    }

    for (int atom = 0; atom < natom; atom++) {
        int atom_naux = naux_per_atom_[atom];

        for (int i = 0; i < nmat; i++) {
            double* raLp = rho_a_L_[atom][i]->pointer()[0];
            double* rtKp = rho_tilde_K_[i]->pointer()[0];

            SharedMatrix JXcopy = J_X_[atom]->clone();
            SharedMatrix rtKbuff = rho_a_L_[atom][i]->clone();
            std::vector<int> ipiv(atom_naux);

            double* JXcp = JXcopy->pointer()[0];
            double* rtKbp = rtKbuff->pointer()[0];

            for (const int& L_a : atom_to_aux_shells_[atom]) {
                int l_off = atom_aux_shell_function_offset_[atom][L_a];
                int l_start = auxiliary_->shell(L_a).start();
                int num_l = auxiliary_->shell(L_a).nfunction();
                double bump_l = atom_aux_shell_bump_value_[atom][L_a];

                for (int dl = 0; dl < num_l; dl++) {
                    rtKbp[l_off + dl] *= bump_l;
                }
            }

            C_DGESV(atom_naux, 1, JXcp, atom_naux, ipiv.data(), rtKbp, atom_naux);

            for (const int& K_a : atom_to_aux_shells_[atom]) {
                int k_off = atom_aux_shell_function_offset_[atom][K_a];
                int k_start = auxiliary_->shell(K_a).start();
                int num_k = auxiliary_->shell(K_a).nfunction();
                double bump_k = atom_aux_shell_bump_value_[atom][K_a];

                for (int dk = 0; dk < num_k; dk++) {
                    rtKp[k_start + dk] += bump_k * rtKbp[k_off + dk];
                }
            }
        }
    }
}

void LocalDFJ::build_J_tilde_L() {

    bool uninitialized = (J_tilde_L_.size() == 0);
    if (uninitialized) J_tilde_L_.resize(rho_tilde_K_.size());

    // TODO: Technically O(N^2), could make it O(N) with a much larger prefactor with CFMMTree
    // Worth investigating in the future
    for (int i = 0; i < rho_tilde_K_.size(); i++) {
        J_tilde_L_[i] = linalg::doublet(Jmet_, rho_tilde_K_[i]);
    }
}

void LocalDFJ::build_J_L(const std::vector<SharedMatrix>& D) {
    bool uninitialized = (J_L_.size() == 0);

    if (uninitialized) {
        J_L_.resize(D.size());
        for (int i = 0; i < D.size(); i++) {
            J_L_[i] = std::make_shared<Matrix>(auxiliary_->nbf(), 1);
        }
    }

    df_cfmm_tree_->df_set_contraction(ContractionType::DF_AUX_PRI);
    df_cfmm_tree_->build_J(ints_, D, J_L_, Jmet_max_);
}

void LocalDFJ::build_I_KX() {
    unsigned natom = molecule_->natom();
    unsigned naux = auxiliary_->nbf();
    unsigned aux_nshell = auxiliary_->nshell();
    int nmat = J_L_.size();

    std::vector<SharedMatrix> dJ_L;
    dJ_L.resize(nmat);
    for (int i = 0; i < nmat; i++) {
        dJ_L[i] = std::make_shared<Matrix>(naux, 1);
        dJ_L[i]->copy(J_L_[i]);
        dJ_L[i]->subtract(J_tilde_L_[i]);
    }

    bool uninitialized = (I_KX_[0].size() == 0);
    for (int atom = 0; atom < natom; atom++) {
        for (int i = 0; i < nmat; i++) {
            if (uninitialized) I_KX_[atom].push_back(std::make_shared<Matrix>(naux_per_atom_[atom], 1));
            I_KX_[atom][i]->zero();
        }
    }

    for (int atom = 0; atom < natom; atom++) {
        int atom_naux = naux_per_atom_[atom];

        for (int i = 0; i < nmat; i++) {
            double* dJLp = dJ_L[i]->pointer()[0];
            double* IKXp = I_KX_[atom][i]->pointer()[0];
            SharedMatrix JXcopy = J_X_[atom]->clone();
            std::vector<int> ipiv(atom_naux);

            double* JXcp = JXcopy->pointer()[0];

            for (const int& L_a : atom_to_aux_shells_[atom]) {
                int l_off = atom_aux_shell_function_offset_[atom][L_a];
                int l_start = auxiliary_->shell(L_a).start();
                int num_l = auxiliary_->shell(L_a).nfunction();
                double bump_l = atom_aux_shell_bump_value_[atom][L_a];

                for (int dl = 0; dl < num_l; dl++) {
                    IKXp[l_off + dl] = bump_l * dJLp[l_start + dl];
                }
            }

            C_DGESV(atom_naux, 1, JXcp, atom_naux, ipiv.data(), IKXp, atom_naux);

            for (const int& K_a : atom_to_aux_shells_[atom]) {
                int k_off = atom_aux_shell_function_offset_[atom][K_a];
                int k_start = auxiliary_->shell(K_a).start();
                int num_k = auxiliary_->shell(K_a).nfunction();
                double bump_k = atom_aux_shell_bump_value_[atom][K_a];

                for (int dk = 0; dk < num_k; dk++) {
                    IKXp[k_off + dk] *= bump_k;
                }
            }
        }
    }

}

void LocalDFJ::build_G_component(const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J) {

    timer_on("LocalDFJ: J");

    for (auto& Jmat : J) {
        Jmat->zero();
    }

    unsigned nmat = D.size();
    unsigned nbf = primary_->nbf();
    unsigned natom = molecule_->natom();
    unsigned aux_nshell = auxiliary_->nshell();
    unsigned pri_nshell = primary_->nshell();

    build_rho_a_L(D);
    build_rho_tilde_K();
    build_J_tilde_L();
    build_J_L(D);
    build_I_KX();

    // Contraction in Equation 23
    df_cfmm_tree_->df_set_contraction(ContractionType::DF_PRI_AUX);
    df_cfmm_tree_->build_J(ints_, rho_tilde_K_, J, Jmet_max_);

    // set up I_KX_max for screening purposes
    SharedMatrix I_KX_max = std::make_shared<Matrix>(natom, aux_nshell);
    I_KX_max->zero();
    auto I_KX_maxp = I_KX_max->pointer();

    if (density_screening_) {
#pragma omp parallel for
        for (size_t atom = 0; atom < natom; atom++) {
            for (const auto K : atom_to_aux_shells_[atom]) { 
                for (size_t i = 0; i < D.size(); i++) {
                    double* IKXp = I_KX_[atom][i]->pointer()[0];

		            int k_off = atom_aux_shell_function_offset_[atom][K];
                    int num_k = auxiliary_->shell(K).nfunction();
                    for (int k = k_off; k < k_off + num_k; k++) {
                        I_KX_maxp[atom][K] = std::max(I_KX_maxp[atom][K], std::abs(IKXp[k]));
                    }
                }
            }
        }
    }

    // Contraction in Equation 25
#pragma omp parallel for
    for (int K = 0; K < aux_nshell; K++) {
        int k_start = auxiliary_->shell(K).start();
        int num_k = auxiliary_->shell(K).nfunction();

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        for (const int& atom : aux_shell_to_atoms_[K]) {
            int k_off = atom_aux_shell_function_offset_[atom][K];

            for (const int& UV_a : atom_to_pri_shells_[atom]) {
                int U = UV_a / pri_nshell;
                int V = UV_a % pri_nshell;

                if (density_screening_) {
    		        double screen_val = I_KX_maxp[atom][K] * I_KX_maxp[atom][K] * Jmet_max_[K] * ints_[thread]->shell_pair_value(U,V);
		            if (screen_val < cutoff_ * cutoff_) continue;
                }

                int u_start = primary_->shell(U).start();
                int num_u = primary_->shell(U).nfunction();

                int v_start = primary_->shell(V).start();
                int num_v = primary_->shell(V).nfunction();

                int prefactor = (U == V) ? 1.0 : 2.0;

                ints_[thread]->compute_shell(K, 0, U, V);
                const double *buffer = ints_[thread]->buffer();

                for (int i = 0; i < nmat; i++) {
                    double* Kuvp = const_cast<double *>(buffer);
                    double* IKXp = I_KX_[atom][i]->pointer()[0];
                    double** Jp = J[i]->pointer();

                    for (int du = 0; du < num_u; du++) {
                        for (int dv = 0; dv < num_v; dv++) {
                            for (int dk = 0; dk < num_k; dk++) {
#pragma omp atomic
                                Jp[u_start + du][v_start + dv] += prefactor * 
                                    Kuvp[dk * num_u * num_v + du * num_v + dv] * IKXp[k_off + dk];
                            }
                        }
                    }
                }
            }
        }
    }

    for (const auto& Jmat : J) {
        Jmat->hermitivitize();
    }

    timer_off("LocalDFJ: J");

}

void LocalDFJ::print_header() {
    if (print_) {
        outfile->Printf("  ==> Direct Local Density Fitted J (with CFMM) <==\n\n");
        outfile->Printf("    Primary Basis: %11s\n", primary_->name().c_str());
        outfile->Printf("    Auxiliary Basis: %11s\n", auxiliary_->name().c_str());
        outfile->Printf("    Max Multipole Order: %11d\n", df_cfmm_tree_->lmax());
        outfile->Printf("    Max Tree Depth: %11d\n", df_cfmm_tree_->nlevels());
        outfile->Printf("\n");
    }
}

}