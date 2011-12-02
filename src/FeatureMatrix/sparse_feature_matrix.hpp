//
//  generic_feature_matrix.h
//  kqp
// 
//  This file contains all the virtual definitions for the classes needed
//
//  Copyright 2011 Benjamin Piwowarski. All rights reserved.
//

#ifndef __KQP__H__
#define __KQP__H__

namespace kqp {
    //! A scalar sparse vector
    template <typename _Scalar>
    class SparseVector {
    public:
        typedef _Scalar Scalar;
    };
    
    /**
     * @brief A class that supports sparse vectors in a (high) dimensional space.
     * @ingroup FeatureMatrix
     */
    template <typename Scalar> 
    class SparseScalarMatrix : public FeatureList<SparseVector<Scalar> > {
        //! The dimension of vectors (0 if no limit)
        Index dimension;
    public:
        SparseScalarMatrix(Index dimension) {
        }
    };
    
    
} // end namespace kqp

#endif