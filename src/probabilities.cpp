#include "probabilities.h"

using namespace kqp;

template <typename scalar, class F> KernelOperator<scalar,F>::KernelOperator(const OperatorBuilder<scalar,F>& evd, bool copy)
{
    mX = copy ? evd.getX() : evd.getX().copy();
    mY.noalias() = (evd.getY() * evd.getZ()).eval();
    mS = copy ? evd.getEigenValues() : evd.getEigenValues().copy();
}




template<typename scalar, class F> size_t KernelOperator<scalar, F>::getRank() const {
    return mY.cols(); 
}




template<typename scalar, class F> Density<scalar,F>::Density(const OperatorBuilder<scalar, F>& evd, bool copy) {
    super(evd, copy);
}


template<typename scalar, class F> double Density<scalar,F>::computeProbability(const Event<scalar, F> &subspace,
                                       bool fuzzyEvent) const {
    Matrix result = getProbabilityMatrix(subspace, fuzzyEvent);

    if (result.rows() == 0) return 0;

    return result.squared_norm();
}


template<typename scalar, class F> typename Density<scalar,F>::Matrix 
Density<scalar,F>::getProbabilityMatrix(const Event<scalar, F> &subspace,
                                           bool fuzzyEvent) const {
    // Compute Y_s^T (P) Y_d S_d
    Matrix mP = subspace.mY.transpose() * subspace.mX.computeInnerProducts(this->mX) * this->mY * this->mS;
        
    // Pre-multiply the result by S_s if using fuzzy subspaces
    if (fuzzyEvent)
        mP = subspace.mS * mP;
    
    return mP;
}



