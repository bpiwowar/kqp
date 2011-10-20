//
//  kqp.cpp
//  kqp
//
//  Created by Benjamin Piwowarski on 26/05/2011.
//  Copyright 2011 University of Glasgow. All rights reserved.
//

#include "kqp.h"

using namespace kqp;

double kqp::EPSILON = 1e-15;


/** 
 
 @mainpage Kernel Quantum Probability project 
 @author B. Piwowarski
 @date May 2011
 
 This project aims at providing an API that allows to compute quantum densities (semi-definite positive hermitian operators) 
 or events (subspaces). It provides tools to compute quantum probabilities and update densities (conditionalisation), and 
 supports the use of kernels to implicitely define the space, thus allowing working in very high dimensional spaces. 

 This project is <a href="http://sourceforge.net/projects/kqp/">hosted on SourceForge</a>.
*/
