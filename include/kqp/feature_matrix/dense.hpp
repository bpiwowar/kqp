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
#ifndef __KQP_DENSE_FEATURE_MATRIX_H__
#define __KQP_DENSE_FEATURE_MATRIX_H__

#include <numeric>
#include <kqp/feature_matrix.hpp>
#include <kqp/subset.hpp>
#include <kqp/intervals.hpp>

namespace kqp {
    
    template <typename Scalar> class DenseMatrix;
    template <typename Scalar> class DenseFeatureSpace;
    
    /**
     * @brief A feature matrix where vectors are dense vectors in a fixed dimension.
     * @ingroup FeatureMatrix
     */
    template <typename Scalar> 
    class DenseMatrix : public FeatureMatrixBase<Scalar> {
    public:       
        KQP_SCALAR_TYPEDEFS(Scalar);
        typedef DenseMatrix<Scalar> Self;
   
   
        ~DenseMatrix() {}
        
        //! Null constructor: will set the dimension with the first feature vector
        DenseMatrix() {}
        
        //! Construct an empty feature matrix of a given dimension
        DenseMatrix(Index dimension) : m_matrix(dimension, 0) {
        }

        //! Construction by copying a dense matrix
        DenseMatrix(const ScalarMatrix &m) : m_matrix(m) {}
        
        //! Copy constructor
        DenseMatrix(const Self &other) : m_gramMatrix(other.m_gramMatrix),  m_matrix(other.m_matrix) {}
        
        //! Creates from a matrix
        static FMatrix create(const ScalarMatrix &m) {
            return FMatrix(new Self(m));
        }
        
#ifndef SWIG
        //! Construction by copying a dense matrix
        DenseMatrix(ScalarMatrix &&m) : m_matrix(m) {}
        
        //! Creates from a matrix
        static FMatrix create(const ScalarMatrix &&m) {
            return FMatrix(new Self(std::move(m)));
        }
#endif

   
        /**
         * Add a vector (from a template expression)
         */
        template<typename Derived>
        void add(const Eigen::DenseBase<Derived> &m) {
            if (m_matrix.cols() == 0) 
                m_matrix.resize(m.rows(), 0);
            else if (m.rows() != m_matrix.rows())
                KQP_THROW_EXCEPTION_F(illegal_operation_exception, 
                                      "Cannot add a vector of dimension %d (dimension is %d)", % m.rows() % m_matrix.rows());

            Index n = m_matrix.cols();
            m_matrix.conservativeResize(m_matrix.rows(), n + m.cols());
            this->m_matrix.block(0, n, m_matrix.rows(), m.cols()) = m; 
        }
        
          /**
         * Add a vector (from a template expression)
         */
        template<typename Derived>
        void add(const Eigen::DenseBase<Derived> &m, const std::vector<bool> &which) {
            if (m_matrix.cols() == 0) 
                m_matrix.resize(m.rows(), 0);
            if (m.rows() != m_matrix.rows())
                KQP_THROW_EXCEPTION_F(illegal_operation_exception, 
                                      "Cannot add a vector of dimension %d (dimension is %d)", % m.rows() % m_matrix.rows());
            
            Index s = std::accumulate(which.begin(), which.end(), 0);
            m_matrix.conservativeResize(m_matrix.rows(), s + m.cols());

            Intervals intervals(which);
            Index offset = m_matrix.cols();
            for(auto i = intervals.begin(); i != intervals.end(); i++) {
                Index cols = i->second - i->first;
                this->m_matrix.block(0, offset, m_matrix.rows(), cols) = m.block(0, i->first, m_matrix.rows(), cols);
                offset += cols;
            }
                                
        }
        
        //! Get a const reference to the m_matrix
        const ScalarMatrix& getMatrix() const {
            return this->m_matrix;
        }
        
        const ScalarMatrix &toDense() const {
            return this->m_matrix;
        }

        inline static const Self &cast(const FMatrixBase &m) {
            return dynamic_cast<const Self&>(m);
        }
        
        void add(const FMatrixBase &other, const std::vector<bool> *which = NULL) override {
    
            if (which) this->add(cast(other).getMatrix(), *which);
            else this->add(cast(other).getMatrix());
        }
        
                       
        virtual Index size() const override { 
            return m_matrix.cols();
        }
        
        Index dimension() const {
            return m_matrix.rows();
        }

        //! Returns the Gram matrix
        const ScalarMatrix &gramMatrix() const {
            if (m_matrix.cols() == m_gramMatrix.rows()) return m_gramMatrix;
            
            // We lose space here, could be used otherwise???
            Index current = m_gramMatrix.rows();
            if (current < m_matrix.cols()) 
                m_gramMatrix.conservativeResize(m_matrix.cols(), m_matrix.cols());
            
            Index tofill = m_matrix.cols() - current;
            
            // Compute the remaining inner products
            m_gramMatrix.bottomRightCorner(tofill, tofill).noalias() = this->getMatrix().rightCols(tofill).adjoint() * this->getMatrix().rightCols(tofill);
            m_gramMatrix.topRightCorner(current, tofill).noalias() = this->getMatrix().leftCols(current).adjoint() * this->getMatrix().rightCols(tofill);
            m_gramMatrix.bottomLeftCorner(tofill, current) = m_gramMatrix.topRightCorner(current, tofill).adjoint().eval();
            
            return m_gramMatrix;
        }
        
        //! Computes the inner product with another m_matrix
        template<class DerivedMatrix>
        void _inner(const Self &other, DerivedMatrix &result) const {
            result = this->getMatrix().adjoint() * other.getMatrix();
        }
        
        
        FMatrixBasePtr linearCombination(const ScalarAltMatrix & mA, Scalar alpha, const Self *mY, const ScalarAltMatrix *mB, Scalar beta) const override {
            ScalarMatrix m(alpha * (getMatrix() * mA));            
            if (mY != 0) 
                m +=  beta * mY->getMatrix() * *mB;
                
            return FMatrixBasePtr(new Self(std::move(m)));   
        }
        

        FMatrixBasePtr subset(const std::vector<bool>::const_iterator &begin, const std::vector<bool>::const_iterator &end) const override {
            ScalarMatrix m;
            select_columns(begin, end, this->m_matrix, m);
            return FMatrixBasePtr(new Self(std::move(m)));
        }
        
        virtual FMatrixBasePtr copy() const override {
            return FMatrixBasePtr(new Self(*this));
        }


               
        virtual FMatrixBase& operator=(const FMatrixBase &other) override {
            m_matrix = dynamic_cast<const Self&>(other).m_matrix;
            m_gramMatrix = dynamic_cast<const Self&>(other).m_matrix;
            return *this;
        }

        
        const ScalarMatrix * operator->() const {
            return &m_matrix;
        }
        const ScalarMatrix & operator*() const {
            return m_matrix;
        }
    private:        
        //! Cache of the gram m_matrix
        mutable ScalarMatrix m_gramMatrix;
        
        //! Our m_matrix
        ScalarMatrix m_matrix;

        friend class DenseFeatureSpace<Scalar>;
    };
    
    
    
    template<typename Scalar>
    std::ostream& operator<<(std::ostream &out, const DenseMatrix<Scalar> &f) {
        return out << "[Dense Matrix with scalar " << KQP_DEMANGLE((Scalar)0) << "]" << std::endl << f.getMatrix();
    }
    
    
    /**
     * The feature space for dense feature m_matrix with canonical kernel
     */
    template<typename Scalar>
    class DenseFeatureSpace : public FeatureSpaceBase<Scalar> {
    public:  
        KQP_SCALAR_TYPEDEFS(Scalar);
        
        static FSpace create(Index dimension) { return FSpace(new DenseFeatureSpace(dimension)); }
        
        DenseFeatureSpace(Index dimension) : m_dimension(dimension) {}
        
        FSpaceBasePtr copy() const override {
            return FSpaceBasePtr(new DenseFeatureSpace(*this));
        }
        
        inline static const DenseMatrix<Scalar>& cast(const FeatureMatrix<Scalar> &mX) { return dynamic_cast<const DenseMatrix<Scalar> &>(*mX); }
        inline static const DenseMatrix<Scalar>& cast(const FMatrixBase &mX) { return dynamic_cast<const DenseMatrix<Scalar> &>(mX); }

        FMatrixBasePtr newMatrix(const ScalarMatrix &mX) const {
            return FMatrixBasePtr(new DenseMatrix<Scalar>(mX));            
        }
        
        
        // Overriden methods

        Index dimension() const override { return m_dimension; }


        virtual FMatrixBasePtr newMatrix() const override {
            return FMatrixBasePtr(new DenseMatrix<Scalar>(m_dimension));
        }
        virtual FMatrixBasePtr newMatrix(const FMatrixBase &mX) const override {
            return FMatrixBasePtr(new DenseMatrix<Scalar>(cast(mX)));            
        }

        virtual bool canLinearlyCombine() const override {
            return true;
        }

        const ScalarMatrix &k(const FeatureMatrixBase<Scalar> &mX) const override {
            return cast(mX).gramMatrix();
        }
        
        virtual ScalarMatrix k(const FeatureMatrixBase<Scalar> &mX1, const ScalarAltMatrix &mY1, const RealAltVector &mD1,
                               const FeatureMatrixBase<Scalar> &mX2, const ScalarAltMatrix &mY2, const RealAltVector &mD2) const override {        
            return mD1.asDiagonal() * mY1.transpose() * cast(mX1)->adjoint() * *cast(mX2) * mY2 * mD2.asDiagonal();
        };
        
        virtual FMatrixBasePtr linearCombination(const FMatrixBase &mX, const ScalarAltMatrix &mA, Scalar alpha, 
                                             const FMatrixBase *mY, const ScalarAltMatrix *mB, Scalar beta) const override {
            return FMatrixBasePtr(cast(mX).linearCombination(mA, alpha, dynamic_cast<const DenseMatrix<Scalar> *>(mY), mB, beta));
        }

    private:
        Index m_dimension;
    };
    
# // Extern templates
# ifndef SWIG
# define KQP_SCALAR_GEN(scalar) extern template class DenseMatrix<scalar>; extern template class DenseFeatureSpace<scalar>;
# include <kqp/for_all_scalar_gen>
# endif
    
} // end namespace kqp

#endif
