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

#ifndef __KQP_CLEANING_COLLAPSE_H__
#define __KQP_CLEANING_COLLAPSE_H__

#include <kqp/kqp.hpp>
#include <kqp/feature_matrix.hpp>
#include <kqp/cleanup.hpp>


namespace kqp {

	template<typename Scalar> 
	class CleanerCollapse: public Cleaner<Scalar> {
		virtual void cleanup(Decomposition<Scalar> &d) const {
			if (!ScalarDefinitions<Scalar>::isIdentity(d.mY)) {
				d.mX = d.fs->linearCombination(d.mX, d.mY);
				d.mY = typename kqp::AltDense<Scalar>::IdentityType(d.mX->size(), d.mX->size());
			}
		};
	};

}

#endif