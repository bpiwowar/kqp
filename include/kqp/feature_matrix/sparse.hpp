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
        
        SparseMatrix() : m_size(0), m_dimension(0) {}
        SparseMatrix(Index dimension) : m_size(0), m_dimension(dimension) {}
        
    protected:
        typedef Eigen::SparseVector<Scalar> SparseVector;
        struct Row {
            SparseVector  vector;
        };
        
        // --- Base methods 
        Index _size() const { 
            return m_size;
        }
        
        Index _dimension() const {
            return m_dimension;
        }
        
        void _add(const Self &other, std::vector<bool> *which = NULL)  {
            std::vector<Index> ix;
            Index toAdd = 0;
            if (which) {
                for(size_t i = 0; i < which->size(); i++)
                    if ((*which)[i]) ix.push_back(i);
            } else toAdd = other._size();
            
            
            auto j = m_matrix.begin();
            
            for(auto it = other.m_matrix.begin(), end = other.m_matrix.end(); it != end; it++) {
                // Search for the right element
                while (j != m_matrix.end() && j->first < it->first) j++;
                
                if (j != m_matrix.end() && j->first != it->first) 
                    j = m_matrix.insert(--j, std::pair<Index,Row>(it->first, Row(it->second)));
                else {
                    // OK, now add elements
                    SparseVector &x = j->second.vector;
                    const SparseVector &y = it->second.vector;
                    KQP_THROW_EXCEPTION(not_implemented_exception, "_add");
                    if (!which) {
                        //                        x.segment(m_size, m_size+y.size()) = y;
                    } else {
                        for(size_t k = 0; k < ix.size() && ix[k] < y.size(); k++) 
                            ;
                        //                            x[m_size+ix[k]] = y[ix[k]];
                    }
                }
                
            }
            
            m_size += toAdd;
        }
        
        const ScalarMatrix &_inner() const {
            // Nothing to do
            if (_size() == 0 || m_size == m_gramMatrix.rows()) return m_gramMatrix;
            
            // Resize the gram matrix
            Index current = m_gramMatrix.rows();
            if (current < m_size) 
                m_gramMatrix.conservativeResize(_size(), _size());
            Index tofill = _size() - current;
            m_gramMatrix.bottomRightCorner(tofill, tofill).setZero();
            m_gramMatrix.topRightCorner(current, tofill).setZero();
            
            // Computes \f$ \sum_i L_i^T * L_i \f$ for the range [current, current+toFill-1]
            // Each row is of the form (a b 0) where a has size current, b is between 0 and toFill
            // so the update is
            // a^T a   a^T b   0
            // b^T a   b^T b   0
            // 0       0       0
            
            // In this loop, we only compute a^t b and b^t b since the rest is already computed (a^T a), obtained by symmetry (b^T a), or is zero
            for(auto i = m_matrix.begin(), end = m_matrix.end(); i != m_matrix.end(); i++) {
                const ScalarVector &x = i->second.vector;
                
                // No need to update for this vector
                if (x.size() < current) continue;
                
                // Update the relevant part
                m_gramMatrix.block(0, current, x.size(), x.size() - current) += x.transpose() * x.segment(current, x.size() - current);
                
            }
            
            // Just to fill the rest
            m_gramMatrix.bottomLeftCorner(tofill, current) = m_gramMatrix.topRightCorner(current, tofill).adjoint().eval();
            
            return m_gramMatrix;
        }
        
        //! Computes the inner product with another matrix
        template<class DerivedMatrix>
        void _inner(const Self &other, const Eigen::MatrixBase<DerivedMatrix> &result) const {
            if (result.rows() != _size() || result.cols() != other.size())
                result.derived().resize(result.rows(), result.cols());
            
            auto j = other.m_matrix.begin(), jEnd = other.m_matrix.end();
            for(auto i = m_matrix.begin(), end = m_matrix.end(); i != m_matrix.end(); i++) {
                // Try to find a match
                while (j != j.end() && j->first < i->first) 
                    j++;
                if (j == jEnd) break;
                
                // Update if the row match
                if (i->first == j->first) {
                    const SparseVector &x = i->second->vector;
                    const SparseVector &y = j->second->vector;
                    result += x.transpose() * y;
                }
            }
        }        
        
        void _subset(const std::vector<bool>::const_iterator &begin, const std::vector<bool>::const_iterator &end, Self &into) const {
        }
        
    private:
        
        //! The Gram matrix
        mutable ScalarMatrix m_gramMatrix;
        
        //! The sparse matrix: a map from row indices to a Row
        std::map<Index, Row> m_matrix;
        
        //! The number of pre-images
        Index m_size;
        
        //! Underlying dimension
        Index m_dimension;
        
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
    
    
# // Extern templates
# define KQP_SCALAR_GEN(scalar) \
   extern template class SparseMatrix<scalar>;
# include <kqp/for_all_scalar_gen>
    
} // end namespace kqp

#endif