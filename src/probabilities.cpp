//
//  probabilities.cpp
//  kqp
//
//  Created by Benjamin Piwowarski on 26/05/2011.
//  Copyright 2011 University of Glasgow. All rights reserved.
//

#include "probabilities.h"

using namespace kqp;

template <typename scalar, class F> KernelOperator<scalar,F>::KernelOperator(const DensityBuilder<scalar,F>& evd, bool copy)
{
    mX = copy ? evd.getX() : evd.getX().copy();
    mY.noalias() = (evd.getY() * evd.getZ()).eval();
    mS = copy ? evd.getEigenValues() : evd.getEigenValues().copy();
}




template<typename scalar, class F> size_t KernelOperator<scalar, F>::getRank() const {
    return mY.cols(); 
}




template<typename scalar, class F> Density<scalar,F>::Density(const DensityBuilder<scalar, F>& evd, bool copy) {
    super(evd, copy);
}


template<typename scalar, class F> double Density<scalar,F>::computeProbability(const Subspace<scalar, F> &subspace,
                                       bool fuzzyEvent) const {
    Matrix result = getProbabilityMatrix(subspace, fuzzyEvent);

    if (result.rows() == 0) return 0;

    return result.squared_norm();
}


template<typename scalar, class F> typename Density<scalar,F>::Matrix 
Density<scalar,F>::getProbabilityMatrix(const Subspace<scalar, F> &subspace,
                                           bool fuzzyEvent) const {
    // Compute Y_s^T (P) Y_d S_d
    Matrix mP = subspace.mY.transpose() * subspace.mX.computeInnerProducts(this->mX) * this->mY * this->mS;
        
    // Pre-multiply the result by S_s if using fuzzy subspaces
    if (fuzzyEvent)
        mP = subspace.mS * mP;
    
    return mP;
}



