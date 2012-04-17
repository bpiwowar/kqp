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
     * In practice, the matrix is a map from a row index to a vector (along with a count of the number of zeros),
     * where each vector has a size less or equal to the number of columns of the sparse matrix.
     * @ingroup FeatureMatrix
     */
    template <typename _Scalar> 
    class SparseDenseMatrix : public FeatureMatrix<SparseDenseMatrix<_Scalar>> {
    public:
        typedef SparseDenseMatrix<_Scalar> Self;
        KQP_FMATRIX_TYPES(SparseDenseMatrix);
        
        
        SparseDenseMatrix() : size(0), dimension(0) {}
        SparseDenseMatrix(Index dimension) : size(0), dimension(dimension) {}
        
        // --- Base methods 
        
        Index _size() const { 
            return size;
        }
        
        Index _dimension() const {
            return dimension;
        }
        
        void _add(const Self &other, std::vector<bool> *which = NULL)  {
            std::vector<Index> ix;
            Index toAdd = 0;
            if (which) {
                for(size_t i = 0; i < which->size(); i++)
                    if ((*which)[i]) ix.push_back(i);
            } else toAdd = other._size();
                
            
            auto j = matrix.begin();
            
            for(auto it = other.matrix.begin(), end = other.matrix.end(); it != end; it++) {
                // Search for the right element
                while (j != matrix.end() && j->first < it->first) j++;

                if (j != matrix.end() && j->first != it->first) 
                    j = matrix.insert(--j, std::pair<Index,Row>(it->first, Row(it->second.first, ScalarVector::Zero(toAdd+size))));
                else {
                    j->second.second.conservativeResize(toAdd+size);
                    j->second.second.bottomRows(toAdd).setZero();
                }
                
                // OK, now add elements
                ScalarVector &x = j->second.second;
                const ScalarVector &y = it->second.second;
                if (!which)
                    x.segment(size, size+y.size()) = y;
                else {
                    for(size_t k = 0; k < ix.size() && ix[k] < y.size(); k++) 
                        x[size+ix[k]] = y[ix[k]];
                }
                j->second.first += it->second.first;
            }
            
            size += toAdd;
            
        }
        
        
        const ScalarMatrix &_inner() const {
            // Nothing to do
            if (_size() == 0 || _size() == gramMatrix.rows()) return gramMatrix;
            
            // Resize the gram matrix
            Index current = gramMatrix.rows();
            if (current < _size()) 
                gramMatrix.conservativeResize(_size(), _size());
            Index tofill = _size() - current;
            gramMatrix.bottomRightCorner(tofill, tofill).setZero();
            gramMatrix.topRightCorner(current, tofill).setZero();
            
            // Computes \f$ \sum_i L_i^T * L_i \f$ for the range [current, current+toFill-1]
            // Each row is of the form (a b 0) where a has size current, b is between 0 and toFill
            // so the update is
            // a^T a   a^T b   0
            // b^T a   b^T b   0
            // 0       0       0
            
            // In this loop, we only compute a^t b and b^t b since the rest is already computed (a^T a), obtained by symmetry (b^T a), or is zero
            for(auto i = matrix.begin(), end = matrix.end(); i != matrix.end(); i++) {
                const ScalarVector &x = i->second.second;
                
                // No need to update for this vector
                if (x.size() < current) continue;
                
                // Update the relevant part
                gramMatrix.block(0, current, x.size(), x.size() - current) += x.transpose() * x.segment(current, x.size() - current);
                
            }
            
            // Just to fill the rest
            gramMatrix.bottomLeftCorner(tofill, current) = gramMatrix.topRightCorner(current, tofill).adjoint().eval();
            
            return gramMatrix;
        }
        
        //! Computes the inner product with another matrix
        template<class DerivedMatrix>
        void _inner(const Self &other, const Eigen::MatrixBase<DerivedMatrix> &result) const {
            if (result.rows() != _size() || result.cols() != other.size())
                result.derived().resize(result.rows(), result.cols());
            
            auto j = other.matrix.begin(), jEnd = other.matrix.end();
            for(auto i = matrix.begin(), end = matrix.end(); i != matrix.end(); i++) {
                // Try to find a match
                while (j != j.end() && j->first < i->first) 
                    j++;
                if (j == jEnd) break;
                
                // Update if the row match
                if (i->first == j->first) {
                    const ScalarVector &x = i->second->second;
                    const ScalarVector &y = j->second->second;
                    result.block(0, 0, x.size(), y.size()) += x.transpose() * y;
                }
            }
        }
        
        
        // Computes alpha * X * A + beta * Y * B (X = *this)
        Self _linear_combination(const ScalarAltMatrix &mA, Scalar alpha, const Self *mY, const ScalarAltMatrix *mB, Scalar beta) const {
            Self r(dimension);
            r.size = mA.cols();
            
            // Compute
            Real max = 0;
            
            for(auto i = matrix.begin(); i != matrix.end(); i++) {
                Row &row = r.matrix[i->first] = Row(i->first, alpha * i->second.second.transpose() * mA.topRows(i->second.second.size()));
                if (!mY)
                    max = std::max(max, row.second.norm());
            }
            
            if (mY && mB) {
                for(auto i = mY->matrix.begin(); i!= mY->matrix.end(); i++) {
                    Row &row = r.matrix[i->first];
                    const ScalarVector &y = i->second.second;
                    if (row.second.size() < y.size()) {
                        Index n = row.second.size();
                        row.second.conservativeResize(y.size());
                        row.second.tail(y.size() - n).setZero();
                    }
                    row.second += beta * y.transpose() * (*mB);
                    max = std::max(max, row.second.norm());
                }
            }
            
            // Check the zeros
            
            
            return r;
        }
        

        
        void _subset(const std::vector<bool>::const_iterator &begin, const std::vector<bool>::const_iterator &end, Self &into) const {
            KQP_THROW_EXCEPTION(not_implemented_exception, "");
        }
        
    private:
        //! Cache of the gram matrix
        mutable ScalarMatrix gramMatrix;
        
        typedef std::pair<Index, ScalarVector> Row;
        //! A map from rows to a pair <non zero count, dense vector>
        std::map<Index, Row> matrix;
        
        //! Number of pre-images
        Index size;
        
        //! Dimension of the space
        Index dimension;
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