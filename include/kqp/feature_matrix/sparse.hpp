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
#include <Eigen/Sparse>

namespace kqp {
    
    
    
    /**
     * @brief A feature matrix where vectors are sparse vectors in a high dimensional space
     *
     * This class makes the hypothesis that vectors have only a few non null components (compared to the dimensionality of the space).
     *
     * @ingroup FeatureMatrix
     */
    template <typename _Scalar> 
    class SparseMatrix : public FeatureMatrix< SparseMatrix<_Scalar> > {
    public:
        KQP_FMATRIX_COMMON_DEFS(SparseMatrix<_Scalar>);
        typedef Eigen::SparseMatrix<Scalar, Eigen::ColMajor> Storage;
        
        
        SparseMatrix()  {}
        SparseMatrix(Index dimension) : m_matrix(dimension, 0) {}
        
#ifndef SWIG
        SparseMatrix(Storage &&storage) : m_matrix(std::move(storage)) {}
#endif

        SparseMatrix(const ScalarMatrix &mat, double threshold) : m_matrix(mat.rows(), mat.cols()) {            
            Matrix<Real, 1, Dynamic> thresholds = threshold * mat.colwise().norm();

            Matrix<Index, 1, Dynamic> countsPerCol((mat.array().abs() >= thresholds.colwise().replicate(mat.rows()).array()).template cast<Index>().colwise().sum());
            
            m_matrix.reserve(countsPerCol);
            
            for(Index i = 0; i < mat.rows(); i++)
                for(Index j = 0; j < mat.cols(); j++)
                    if (std::abs(mat(i,j)) > thresholds[j]) 
                        m_matrix.insert(i,j) = mat(i,j);
        }
        
        SparseMatrix(const Storage &storage) : m_matrix(storage) {}
        SparseMatrix(const Eigen::SparseMatrix<Scalar, Eigen::RowMajor>  &storage) : m_matrix(storage) {}

        ScalarMatrix toDense() {
            return ScalarMatrix(m_matrix);
        }
        
    protected:
        // --- Base methods 
        inline Index _size() const { 
            return m_matrix.cols();
        }
        
        inline Index _dimension() const {
            return m_matrix.rows();
        }
        
        void _add(const Self &other, const std::vector<bool> *which = NULL)  {
            // Computes the indices of the vectors to add
            std::vector<Index> ix;
            Index toAdd = 0;
            if (which) {
                for(size_t i = 0; i < which->size(); i++)
                    if ((*which)[i]) ix.push_back(i);
            } 
            else toAdd = other._size();
            
        }
        
        const ScalarMatrix &_inner() const {
            Index current = m_gramMatrix.rows();
            if (size() == current) return m_gramMatrix;
            
            if (current < size()) 
                m_gramMatrix.conservativeResize(size(), size());
            Index tofill = size() - current;
            
            // Compute the remaining inner products
            Index ncols = m_matrix.cols();
            
            for(Index i = 0; i < ncols; ++i)
                for(Index j = current; j < ncols; ++j)
                    m_gramMatrix(i,j) = m_matrix.col(i).dot(m_matrix.col(j));
            m_gramMatrix.bottomLeftCorner(tofill, current) = m_gramMatrix.topRightCorner(current, tofill).adjoint().eval();
            
            return m_gramMatrix;
        }

        
        //! Computes the inner product with another matrix
        template<class DerivedMatrix>
        void _inner(const Self &other, const Eigen::MatrixBase<DerivedMatrix> &result) const {
            Storage r = this->m_matrix.transpose() * other.m_matrix;
            const_cast<DerivedMatrix&>(result.derived()) = ScalarMatrix(r);
        }        
        
        void _subset(const std::vector<bool>::const_iterator &begin, const std::vector<bool>::const_iterator &end, Self &into) const {
            // Construct
            std::vector<Index> selected;
            auto it = begin;
            for(Index i = 0; i < m_matrix.cols(); i++) {
                if (it == end || *it)
                    selected.push_back(i);
                it++;
            }
            
            // Prepare the resultant sparse matrix
            Storage s(m_matrix.rows(), selected.size());
            Eigen::VectorXi counts(selected.size());
            for(size_t i = 0; i < selected.size(); ++i)
                counts[i] = m_matrix.col(selected[i]).nonZeros();
            s.reserve(counts);
            
            // Fill the result
            for(size_t i = 0; i < selected.size(); ++i)
                for (typename Storage::InnerIterator it(m_matrix,selected[i]); it; ++it) 
                    s.insert(it.row(), i) = it.value();
            
            into = Self(s);
        }
        
        // Computes alpha * X * A + beta * Y * B (X = *this)
        Self _linear_combination(const ScalarAltMatrix &, Scalar , const Self *, const ScalarAltMatrix *, Scalar ) const {
            KQP_THROW_EXCEPTION(illegal_operation_exception, "Cannot compute the linear combination of a Sparse matrix");
        }
        
    private:
        
        //! The Gram matrix
        mutable ScalarMatrix m_gramMatrix;
        

        //! The underlying sparse matrix
        Storage m_matrix;
        
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
    

#ifndef SWIG    
# // Extern templates
# define KQP_SCALAR_GEN(scalar) \
   extern template class SparseMatrix<scalar>;
# include <kqp/for_all_scalar_gen>
#endif

} // end namespace kqp

#endif