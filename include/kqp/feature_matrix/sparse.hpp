/*
 This file is part of the Kernel Quantum Probability library (KQP).
 
 KQP is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 KQP is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with KQP.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __KQP_SPARSE_FEATURE_MATRIX_H__
#define __KQP_SPARSE_FEATURE_MATRIX_H__

#include <kqp/subset.hpp>
#include <kqp/feature_matrix.hpp>

namespace kqp {

    // ---- Sparse matrix in the Dense canonical basis ----
    /**
     * @brief A feature matrix where vectors are in a dense subspace (in the canonical basis).
     *
     * This class makes the hypothesis that vectors have only a few non null components (compared to the dimensionality of the space), 
     * and that those components are mostly the same.
     *
     * @ingroup FeatureMatrix
     */
    template <typename _Scalar> 
    class SparseDenseMatrix : public FeatureMatrix<SparseDenseMatrix<_Scalar>> {
    public:
        KQP_FMATRIX_TYPES(SparseDenseMatrix);
    };
      
    
    // The scalar for dense feature matrices
    template <typename _Scalar> struct FeatureMatrixTypes<SparseDenseMatrix<_Scalar> > {
        typedef _Scalar Scalar;
        enum {
            can_linearly_combine = 1
        };
    };
    
    
    /**
     * @brief A feature matrix where vectors are sparse vectors in a high dimensional space
     *
     * This class makes the hypothesis that vectors have only a few non null components (compared to the dimensionality of the space).
     *
     * @ingroup FeatureMatrix
     */
    template <typename _Scalar> 
    class SparseMatrix : public FeatureMatrix<SparseMatrix<_Scalar>> {
    public:
        KQP_FMATRIX_TYPES(SparseMatrix);
    };
    
    
    
    template<typename Scalar>
    std::ostream& operator<<(std::ostream &out, const SparseMatrix<Scalar> &f) {
        return out << "[Sparse Matrix with scalar " << KQP_DEMANGLE((Scalar)0) << "]" << std::endl << f.getMatrix();
    }
    
    
    // The scalar for dense feature matrices
    template <typename _Scalar> struct FeatureMatrixTypes<SparseMatrix<_Scalar> > {
        typedef _Scalar Scalar;
        enum {
            can_linearly_combine = 0
        };
    };
    
    
    // Extern templates
#define KQP_SCALAR_GEN(scalar) \
    extern template class SparseMatrix<scalar>; \
    extern template class SparseDenseMatrix<scalar>;
#include <kqp/for_all_scalar_gen>
    
} // end namespace kqp

#endif