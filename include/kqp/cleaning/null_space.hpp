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

#ifndef __KQP_REDUCED_SET_NULL_SPACE_H__
#define __KQP_REDUCED_SET_NULL_SPACE_H__

#include <algorithm> // sort

#include <kqp/kqp.hpp>
#include <Eigen/Eigenvalues>

#include <kqp/feature_matrix.hpp>
#include <kqp/subset.hpp>
#include <kqp/cleaning/unused.hpp>

namespace kqp {
#   include <kqp/define_header_logger.hpp>
    DEFINE_KQP_HLOGGER("kqp.cleaning.null_space");

#ifndef SWIG
    template <class ComparableArray, typename Index = int>
    struct IndirectSort {
        const ComparableArray &array;
        IndirectSort(const ComparableArray &array) : array(array) {}
        bool operator() (int i,int j) { return (array[i] < array[j]);}
    };


    template <class ComparableArray, typename Index = int>
    struct AbsIndirectSort {
        const ComparableArray &array;
        AbsIndirectSort(const ComparableArray &array) : array(array) {}
        bool operator() (int i,int j) { return std::abs(array[i]) < std::abs(array[j]);}
    };

    
    template<typename Scalar>
    struct ReducedSetNullSpaceResult {
        KQP_SCALAR_TYPEDEFS(Scalar);
        FMatrixPtr mX;
        ScalarAltMatrix mY;
    };
    
    template<typename Scalar>
    class ReducedSetNullSpace {
    public:
        KQP_SCALAR_TYPEDEFS(Scalar);
    private:
        double m_epsilon;
        std::pair<float,float> m_preImageRatios;            
        std::pair<Index,Index> m_preImagesRange;            
            
    public:
        ReducedSetNullSpace() : 
            m_epsilon(Eigen::NumTraits<Real>::epsilon()),
            m_preImageRatios(std::make_pair(0, std::numeric_limits<Real>::infinity())),
            m_preImagesRange(std::make_pair(0, std::numeric_limits<Index>::max()))
        {}
        
        ReducedSetNullSpace & epsilon(double _epsilon) { this->m_epsilon = _epsilon; return *this; }
        
        //! Set constraints on the number of pre-images
        void setPreImagesPerRank(float reset, float maximum) {
            this->m_preImageRatios = std::make_pair(reset, maximum);
        }

        void setRankRange(Index reset, Index maximum) {
            this->m_preImagesRange = std::make_pair(reset, maximum);
        }        
        /**
         * @brief Removes pre-images with the null space method
         * 
         * Removes pre-images using the null space method.
         * We have \f$ X Z  = 0\f$ and want to find \f$X^\prime\f$ and \f$A\f$ such that
         * \f$ X = (\begin{array}{cc} X^\prime & X^\prime A \end{array} )
         * 
         * @param mF (\b in) the feature matrix \f$X\f$ (\b out) the reduced feature matrix \f$X^\prime\f$
         * @param kernel (in) the null space basis \f$Z\f$ for the matrix \ref mF (out) the matrix \f$A\f$ such that \f$ X^\prime A = X^{\prime\prime} \f$
         * @param mP (out) Permutation matrix so that \f$X P = (X^\prime X^{\prime\prime})\f$
         * @param weights give an order to the different pre-images
         * @param delta
         */
        FMatrixPtr remove(const FMatrixCPtr &mF, ScalarMatrix &kernel, Eigen::PermutationMatrix<Dynamic, Dynamic, Index>& mP, const RealVector &weights) const {
            typedef typename Eigen::PermutationMatrix<Dynamic, Dynamic, Index> Permutation;

            // FIXME: should be wiser
            double delta = 1e-4;
            
            // --- Check if we have something to do
            if (mF->size() == 0)
                return mF->copy();
            
            // --- Look up at the indices to remove
            
            // Get the pre-images available for removal (one per vector of the null space, i.e. by columns in kernel)
            // Summarize the result per pre-image (number of vectors where it appears)
            // Eigen::Matrix<int,Dynamic,1> num_vectors = (kernel.array().abs() > kernel.array().abs().colwise().maxCoeff().replicate(kernel.rows(), 1) * delta).template cast<int>().rowwise().sum();
            
            std::vector<Index> to_remove;
            for(Index i = 0; i < /*num_vectors*/kernel.rows(); i++)
                    to_remove.push_back(i);

            // Sort        
            const Index pre_images_count = kernel.rows();
            const std::size_t remove_size = kernel.cols();
            const Index keep_size = mF->size() - remove_size;
            
            mP = Permutation(mF->size());
            mP.setIdentity();
            
            std::sort(to_remove.begin(), to_remove.end(), IndirectSort<RealVector>(weights));
            std::vector<bool> selection(mF->size(), true);
            std::vector<bool> used(to_remove.size(), false);
            
            ScalarVector v;
            
            // --- Remove the vectors one by one (Pivoted Gauss)
            Index last = mF->size() - 1;

            Index remaining = remove_size;

            while (remaining > 0) {
                // Select the pre-image to remove along with the null space vector
                // that will be used
                // TODO: find a better way

                Eigen::Matrix<Real,Dynamic,1> colnorms = kernel.colwise().norm();

                size_t i = 0;
                Index j = -1;
                Real max = 0;
                Real threshold;
                for(size_t index = 0; index < to_remove.size(); index++) {
                    // remove the ith pre-image
                    i = to_remove[index];
                    if (!selection[i]) continue;
                    
                    // Searching for the highest magnitude
                    j = -1;
                    // ... but we want it above a given threshold
                    max = 0;
                    for(Index k = 0; k < kernel.cols(); k++) {
                        Real x = kernel.array().abs()(i, k);
                        if (!used[k] && x > max && x > delta * colnorms(k)) {
                            threshold = delta * colnorms(k); // just for debug
                            max = x;
                            j = k;
                        }
                    }
                    if (j >= 0) break;
                }
                if (j == -1)
                    KQP_THROW_EXCEPTION_F(assertion_exception,
                     "Could not find a way to remove the a pre-image with null space (%d/%d pre-images). %d remaining.",
                     %kernel.cols() %kernel.rows() %remaining);
                KQP_HLOG_DEBUG_F("Selected pre-image %d with basis vector %d [%g > %g; norm=%g]", %i %j %max %threshold %colnorms(j));

                remaining--;
                used[j] = true;
                
                // Update permutation by putting this vector at the end
                selection[i] = false;            
                mP.indices()(i) = j + keep_size;
                last -= 1;
                
                // Update the matrix
                Scalar kij = kernel(i,j);
                KQP_HLOG_DEBUG_F("Normalizing column %d [norm %g] with the inverse of %g", %j %kernel.col(j).norm() %kij)
                v = kernel.col(j) / kij;
                kernel.col(j) /= -kij;
                
                assert(!kqp::isNaN(kernel.col(j).squaredNorm()));
                
                kernel(i,j) = 0;
                
                kernel = ((Eigen::Identity<Scalar>(pre_images_count, pre_images_count) 
                           - v * ScalarVector::Unit(pre_images_count, i).adjoint()) * kernel).eval();
            }
            
            
            // --- Remove the vectors from mF and set the permutation matrix
            
            // Remove unuseful vectors
            
            select_rows(selection.begin(), selection.end(), kernel, kernel);
            
            // Complete the permutation matrix
            Index count = 0;
            for(size_t index = 0; index < selection.size(); index++) 
                if (selection[index]) {
                    mP.indices()(index) = count++;
                }
            
            return mF->subset(selection.begin(), selection.end());
        }
                
        /**
         * @brief Removes unuseful pre-images 
         *
         * 1. Removes unused pre-images 
         * 2. Computes a \f$LDL^\dagger\f$ decomposition of the Gram matrix to find redundant pre-images
         * 3. Removes newly unused pre-images
         */
        ReducedSetNullSpaceResult<Scalar> run(const FSpaceCPtr &fs, const FMatrixCPtr &mF, const ScalarAltMatrix &mY) const {
            ReducedSetNullSpaceResult<Scalar> result;
            result.mX = mF->copy();
            result.mY = mY;

            auto & _mX = result.mX;
            auto & _mY = result.mY;

			// Dimension of the problem
            const Index N = _mY.rows();
			const Index rank = _mY.cols();
            assert(N == _mX->size());

			// Select with max rank and threshold
            Index target = N;
            if (N > this->m_preImageRatios.second * (Real)rank)
                target = std::min(target, (Index)(this->m_preImageRatios.first * (Real)rank));
			
            if (N > this->m_preImagesRange.second)
                target = std::min(target, this->m_preImagesRange.first);

			// Don't run the EVD if possible
			if (N == target && m_epsilon == 0)
				return std::move(result);
			
            // Removes unused pre-images
            CleanerUnused<Scalar>::run(_mX, _mY);
            
            // EVD
            Eigen::SelfAdjointEigenSolver<ScalarMatrix> evd(fs->k(_mX)); 
            const Eigen::Matrix<Real,Dynamic,1> &d = evd.eigenvalues();

            std::vector<Index> list(N);
            for(Index i = 0; i < N; i++)
                list[i] = i;
            std::sort(list.begin(), list.end(), AbsIndirectSort< Eigen::Matrix<Real,Dynamic,1>, Index >(d));
                

            Real threshold = m_epsilon * (Real)d.size() *  std::abs(d[list.back()]);
            Index nullSize = N - target;
            while (nullSize < N && std::abs(d[nullSize]) < threshold) {
                nullSize++;
            }
			
            KQP_HLOG_DEBUG_F("Rank of used null space is %d (image %d)", %nullSize %(N-nullSize));
			
			assert(nullSize >= 0);
            if (nullSize <= 0)
                return std::move(result);

			// True if the null space is not really null
			bool forced = std::abs(d[nullSize-1]) > threshold;
			KQP_HLOG_DEBUG_F("Last eigenvalue is %g (threshold %g): forced %d", %std::abs(d[nullSize-1]) %threshold %forced);

            // Compute the kernel
            ScalarMatrix kernel(_mX->size(), nullSize);
            for(Index i = 0; i < nullSize; i++)
                kernel.col(i) = evd.eigenvectors().col(list[i]);

            // Remove pre-images using the kernel
            RealVector weights = _mY.rowwise().squaredNorm().array() * fs->k(_mX).diagonal().array().abs();
            Eigen::PermutationMatrix<Dynamic, Dynamic, Index> mP;
            _mX = remove(_mX, kernel, mP, weights);
            
			if (forced) {
				// Compute _mY so that _mX * _mY is orthonormal, ie _mY' _mX' _mX _mY is the identity
				Eigen::SelfAdjointEigenSolver<ScalarMatrix> evd(fs->k(_mX).template selfadjointView<Eigen::Lower>());
				ScalarMatrix __mY = std::move(evd.eigenvectors());
				__mY *= evd.eigenvalues().cwiseAbs().cwiseSqrt().cwiseInverse().asDiagonal();
				__mY *= fs->k(_mX, __mY, mF, mY);
				_mY = __mY;
			} else {
				// Y <- (Id A) P Y
				ScalarMatrix mY2(_mY);
				mY2= (mP * mY2).topRows(_mX->size()) + kernel * (mP * mY2).bottomRows(_mY.rows() - _mX->size());
				
				_mY.swap(mY2);
            }
            // Removes unused pre-images

            CleanerUnused<Scalar>::run(_mX, _mY);
			KQP_HLOG_DEBUG_F("Finished null space cleaning: number of pre-images is %d", %_mX->size());
            return std::move(result);
        }
                
    };

#endif

    template<typename Scalar> class CleanerNullSpace: public Cleaner<Scalar> {
        ReducedSetNullSpace<Scalar> m_cleaner;
    public:
        typedef typename Eigen::NumTraits<Scalar>::Real Real;
        
        CleanerNullSpace() {
        }
        
        void epsilon(double _epsilon) {
			m_cleaner.epsilon(_epsilon);
		}

        void setPreImagesPerRank(float reset, float maximum) {
			if (reset < 0)
				KQP_THROW_EXCEPTION_F(kqp::illegal_argument_exception, "Reset rank (%g) is below 0", %reset);
			if (maximum < reset)
				KQP_THROW_EXCEPTION_F(kqp::illegal_argument_exception, "Maximum rank (%g) is out of bounds (should be greater than reset %g)", %maximum %reset);
            m_cleaner.setPreImagesPerRank(reset, maximum);
        }

        void setRankRange(Index reset, Index maximum) {
			if (reset < 0)
				KQP_THROW_EXCEPTION_F(kqp::illegal_argument_exception, "Reset rank (%d) is below 0", %reset);
			if (maximum < reset)
				KQP_THROW_EXCEPTION_F(kqp::illegal_argument_exception, "Maximum rank (%d) is out of bounds (should be greater than reset %d)", %maximum %reset);
            m_cleaner.setRankRange(reset, maximum);
        }        
        
        virtual void cleanup(Decomposition<Scalar> &d) const {
           auto result = m_cleaner.run(d.fs, d.mX, d.mY); 
           d.mX = std::move(result.mX);
           d.mY = std::move(result.mY);
        }        
    private:
        Real m_epsilon;
    };
    
}

#endif
