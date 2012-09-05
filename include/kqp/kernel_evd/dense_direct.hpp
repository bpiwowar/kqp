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

#ifndef __KQP_DENSE_DIRECT_BUILDER_H__
#define __KQP_DENSE_DIRECT_BUILDER_H__

#include <boost/shared_ptr.hpp>

#include <kqp/kqp.hpp>


#include <Eigen/Eigenvalues>

#include <kqp/kernel_evd.hpp>
#include <kqp/feature_matrix/dense.hpp>
#include <kqp/evd_utils.hpp>

#include <cassert>

namespace kqp {

#ifndef SWIG
    /** Rank update with an Eigen matrix expression (used for recursion termination) */
    template<typename MatrixType, unsigned int UpLo, typename Derived>
    void rankUpdate2(Eigen::SelfAdjointView<MatrixType, UpLo> &&matrix, const Eigen::MatrixBase<Derived> &mA, const typename MatrixType::Scalar alpha) {
        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> mB = mA;
        matrix.rankUpdate(mB, alpha);
    }

    /** 
    * Rank update with an AltMatrix 
    * @param matrix The matrix to be updated
    * @param mA The matrix used for update
    * @param alpha The update coefficient
    */
    template<typename MatrixType, unsigned int UpLo, typename Derived> 
    void rankUpdate2(Eigen::SelfAdjointView<MatrixType, UpLo> &&matrix, const AltMatrixBase<Derived> &mA, const typename MatrixType::Scalar alpha) {
        std::cerr << "Rank update recursion...\n";
        if (mA.derived().isT1())
            rankUpdate2(std::move(matrix), mA.derived().t1(), alpha);
        else 
            rankUpdate2(std::move(matrix), mA.derived().t2(), alpha);
    }
#endif

    /**
     * @brief Direct computation of the density (i.e. matrix representation) for dense vectors.
     * 
     * @ingroup KernelEVD
     */
    template <class Scalar> class DenseDirectBuilder : public KernelEVD<Scalar> {
    public:
        KQP_SCALAR_TYPEDEFS(Scalar);
        typedef Dense<Scalar> FDense;
        
        DenseDirectBuilder(int dimension) : KernelEVD<Scalar>(DenseSpace<Scalar>::create(dimension)), matrix(dimension, dimension) {
            reset();
        }
        
        virtual ~DenseDirectBuilder() {}

        void reset() {
            matrix.setConstant(0);
            KernelEVD<Scalar>::reset();
        }

    protected:


        virtual void _add(Real alpha, const FMatrix &mX, const ScalarAltMatrix &mA) override {
            const auto &_mX = dynamic_cast<const FDense &>(*mX).getMatrix();
            std::cerr << boost::format("mX [%d, %d]\nmA [%d,%d]\n") % _mX.rows() % _mX.cols() % mA.rows() % mA.cols() << std::endl;  
            std::cerr << "Updating of rank " << mX.size() << " in dimension " << matrix.rows() << " x " << matrix.cols() << "...\n";
            // auto a = dynamic_cast<const FDense &>(*mX).getMatrix() * mA;
            // std::cerr << " -- " << a.rows() << " x " << a.cols() << std::endl;
            rankUpdate2(matrix.template selfadjointView<Eigen::Lower>(), dynamic_cast<const FDense &>(*mX).getMatrix() * mA, (Scalar)alpha);
            std::cerr << "Finish rank update...\n";
        }
        
        virtual Decomposition<Scalar> _getDecomposition() const override {
            Decomposition<Scalar> d(this->getFSpace());
            Eigen::SelfAdjointEigenSolver<ScalarMatrix> evd(matrix.template selfadjointView<Eigen::Lower>());
            
            ScalarAltMatrix _mX;
            kqp::ThinEVD<ScalarMatrix>::run(evd, _mX, d.mD);              
            
            d.mX = FMatrix(new Dense<Scalar>(std::move(ScalarMatrix(_mX))));
            d.mY = Eigen::Identity<Scalar>(d.mX.size(), d.mX.size());
            return d;
        }
        
        
    public:
        
        ScalarMatrix matrix;
    };
    
#ifndef SWIG
#define KQP_SCALAR_GEN(scalar) extern template class DenseDirectBuilder<scalar>;
#include <kqp/for_all_scalar_gen.h.inc>
#endif
    
} // end namespace kqp

#endif
