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
    template <typename Scalar> 
    class Sparse : public FeatureMatrixBase<Scalar> {
    public:
        KQP_SCALAR_TYPEDEFS(Scalar);
        typedef Sparse<Scalar> Self;
        typedef Eigen::SparseMatrix<Scalar, Eigen::ColMajor> Storage;
        
        virtual ~Sparse() {}
        
        Sparse()  {}
        Sparse(Index dimension) : m_matrix(dimension, 0) {}
        Sparse(const Self &other) : m_matrix(other.m_matrix) {}
        
#ifndef SWIG
        Sparse(Storage &&storage) : m_matrix(std::move(storage)) {}
        Sparse(Self &&other) : m_matrix(std::move(other.m_matrix)) {}
        
#endif

        Sparse(const ScalarMatrix &mat, double threshold = 0) : m_matrix(mat.rows(), mat.cols()) {            
            Matrix<Real, 1, Dynamic> thresholds = threshold * mat.colwise().norm();

            Matrix<Index, 1, Dynamic> countsPerCol((mat.array().abs() >= thresholds.colwise().replicate(mat.rows()).array()).template cast<Index>().colwise().sum());
            
            m_matrix.reserve(countsPerCol);
            
            for(Index i = 0; i < mat.rows(); i++)
                for(Index j = 0; j < mat.cols(); j++)
                    if (std::abs(mat(i,j)) > thresholds[j]) 
                        m_matrix.insert(i,j) = mat(i,j);
        }
        
        Sparse(const Storage &storage) : m_matrix(storage) {}
        Sparse(const Eigen::SparseMatrix<Scalar, Eigen::RowMajor>  &storage) : m_matrix(storage) {}

        ScalarMatrix toDense() {
            return ScalarMatrix(m_matrix);
        }
        
        // --- Base methods 
        inline Index size() const { 
            return m_matrix.cols();
        }
        
        inline Index dimension() const {
            return m_matrix.rows();
        }
        
        void add(const FMatrixBase &_other, const std::vector<bool> *which = NULL) override {
            const Self &other = dynamic_cast<const Self&>(_other);
            
            // Computes the indices of the vectors to add
            std::vector<Index> ix;
            Index toAdd = 0;
            if (which) {
                for(size_t i = 0; i < which->size(); i++)
                    if ((*which)[i]) 
                        ix.push_back(i);
                    
                toAdd = ix.size();
            } 
            else toAdd = other.size();
            
            //FIXME: re-implement when conservativeResize is implemented in Eigen::SparseMatrix
            
            // Construct the vector of non zero entries count
        
            std::vector<Index> counts;
            for(Index i = 0; i < m_matrix.cols(); i++) 
              counts.push_back(m_matrix.col(i).nonZeros());
        
            for(Index i = 0; i < toAdd; i++)
              counts.push_back(other.m_matrix.col(ix.empty() ? i : ix[i]).nonZeros());
            
        
            // Prepare the resultant sparse matrix
        
            // FIXME: m_matrix.conservativeResize(m_matrix.rows(), m_matrix.cols() + toAdd)
            Index offset = m_matrix.cols();
            Storage s(m_matrix.rows(), m_matrix.cols() + toAdd);
            s.reserve(counts);
            
            // Fill the result
        
            for(Index i = 0; i < m_matrix.cols(); ++i) // FIXME: remove this
                for (typename Storage::InnerIterator it(m_matrix,i); it; ++it) 
                    s.insert(it.row(), i) = it.value();
                    
            for(Index i = 0; i < toAdd; ++i)
                for (typename Storage::InnerIterator it(other.m_matrix, ix.empty() ? i : ix[i]); it; ++it) 
                    s.insert(it.row(), offset+i) = it.value();
                
                    
            // Compress (should not do anything but free the non zero buffer)
            m_matrix = std::move(s);
            m_matrix.makeCompressed();
        }
        
        const ScalarMatrix &gramMatrix() const {
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

        
        FMatrixBasePtr subset(const std::vector<bool>::const_iterator &begin, const std::vector<bool>::const_iterator &end) const override{
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
            
            return FMatrixBasePtr(new Self(std::move(s)));
        }
        
        // Computes alpha * X * A + beta * Y * B (X = *this)
        
        FMatrixBasePtr copy() const override {
            return FMatrixBasePtr(new Self(*this));
        }
        
        //! Direct access to underlying matrix
        const Storage *operator->() const { return &m_matrix; }
        const Storage &operator*() const { return m_matrix; }

        FMatrixBase& operator=(const Self &other)  {
            m_matrix = other.m_matrix;
            m_gramMatrix = other.m_gramMatrix;
            return *this;
        }

        virtual FMatrixBase& operator=(const FMatrixBase &other) override {
            return *this = dynamic_cast<const Self&>(other);
        }

        //! Get the dense matrix
        const Storage &getMatrix() const {
            return m_matrix;
        }


    private:
        
        //! The Gram matrix
        mutable ScalarMatrix m_gramMatrix;
        

        //! The underlying sparse matrix
        Storage m_matrix;
        
    };
    
    
    
    template<typename Scalar>
    std::ostream& operator<<(std::ostream &out, const Sparse<Scalar> &f) {
        return out << "[Sparse Matrix with scalar " << KQP_DEMANGLE((Scalar)0) << "]" << std::endl << f.getMatrix();
    }
    
    
    template<typename Scalar>
    class SparseSpace : public SpaceBase<Scalar> {
    public:  
        KQP_SCALAR_TYPEDEFS(Scalar);
#ifndef SWIG
        using SpaceBase<Scalar>::k;
#endif        
        static FSpace create(Index dimension) { return FSpace(new SparseSpace(dimension)); }
        
        SparseSpace(Index dimension) : m_dimension(dimension) {}
        
        inline static const Sparse<Scalar>& cast(const FeatureMatrixBase<Scalar> &mX) { return dynamic_cast<const Sparse<Scalar> &>(mX); }
        
        Index dimension() const override { return m_dimension; }
        
        virtual FSpaceBasePtr copy() const override { return FSpaceBasePtr(new SparseSpace(m_dimension)); }
        
        virtual FMatrixBasePtr newMatrix() const override {
            return FMatrixBasePtr(new Sparse<Scalar>(m_dimension));
        }
        virtual FMatrixBasePtr newMatrix(const FMatrixBase &mX) const override {
            return FMatrixBasePtr(new Sparse<Scalar>(cast(mX)));            
        }
        
        
        const ScalarMatrix &k(const FeatureMatrixBase<Scalar> &mX) const override {
            return cast(mX).gramMatrix();
        }
        
        virtual ScalarMatrix k(const FeatureMatrixBase<Scalar> &mX1, const ScalarAltMatrix &mY1, const RealAltVector &mD1,
                               const FeatureMatrixBase<Scalar> &mX2, const ScalarAltMatrix &mY2, const RealAltVector &mD2) const override {        
            return mD1.asDiagonal() * mY1.adjoint() * cast(mX1)->adjoint() * *cast(mX2) * mY2 * mD2.asDiagonal();
        };
        
        
    private:
        Index m_dimension;
    
    };
    

#ifndef SWIG    
# // Extern templates
# define KQP_SCALAR_GEN(scalar) extern template class Sparse<scalar>; extern template class SparseSpace<scalar>;
# include <kqp/for_all_scalar_gen.h.inc>
#endif

} // end namespace kqp

#endif