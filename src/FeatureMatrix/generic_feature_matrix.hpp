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
    
    /**
     * @brief A class that supposes that feature vectors know how to compute their inner product, i.e. inner(a,b)
     *        is defined.
     * @ingroup FeatureMatrix
     */
    template <class FVector> 
    class FeatureList : public FeatureMatrix<FVector> {
    public:
        
        Index size() const { return this->list.size(); }
        
        const FVector& get(Index i) const { return this->list[i]; }
        
        virtual void add(const FVector &f) {
            this->list.push_back(f);
        }
        
    private:
        std::vector<FVector> list;
    };
    
} // end namespace kqp

#endif