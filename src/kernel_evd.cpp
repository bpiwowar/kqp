//
//  kernel_evd.cpp
//  kqp
//
//  Created by Benjamin Piwowarski on 23/05/2011.
//  Copyright 2011 University of Glasgow. All rights reserved.
//

#include "kernel_evd.h"

using namespace kqp;

double EPSILON = 1e-17;

template <typename scalar, class F>
typename FeatureMatrix<scalar,F>::Matrix FeatureMatrix<scalar, F>::computeInnerProducts(const FeatureMatrix<scalar, F>& other) const {
        
    Eigen::MatrixXd m = Eigen::MatrixXd(size(), other.size());
    
    for (int i = 0; i < size(); i++)
        for (int j = i; j < other.size(); j++)
            m(j,i) = m(i, j) = computeInnerProduct(i, get(j));

    return m;
}


template <typename scalar, class F>
typename FeatureMatrix<scalar,F>::Vector FeatureMatrix<scalar, F>::computeInnerProducts(const F& vector) const {
    typename FeatureMatrix::Vector innerProducts(size());
    for (int i = size(); --i >= 0;)
        innerProducts[i] = computeInnerProduct(i, vector);
    return innerProducts;
}


template <typename scalar, class F>
typename FeatureMatrix<scalar,F>::Matrix  FeatureMatrix<scalar, F>::computeGramMatrix() const {
    // We loose space here, could be used otherwise???
    Eigen::MatrixXd m = Eigen::MatrixXd(size(), size()).selfadjointView<Eigen::Upper>();
    
    for (int i = size(); --i >= 0;)
        for (int j = i + 1; --j >= 0;) {
            double x = computeInnerProduct(i, get(j));
            m(i,j) = m(j,i) = x;
        }
    return m;
}

// --- Scalar matrix

template <typename scalar> ScalarMatrix<scalar>::~ScalarMatrix() {}


// --- Direct builder

template <typename scalar, class F> void DirectBuilder<scalar,F>::add(double alpha, const F &v) {
    this->list.add(v);
}