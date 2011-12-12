#include <algorithm>
#include <bitset>
#include <vector>
#include <sstream>
#include <cmath>
#include <limits.h>
#include <exception>
#include <complex>

#include "kqp.hpp"
#include "evd_update.hpp"

DEFINE_LOGGER(logger, "kqp.evd-update");

namespace kqp {
    
    // Useful scalar dependant functions
    
    inline bool is_real(double f) { return true; } 
    inline bool is_real(float f) { return true; } 
    template <typename scalar> inline bool is_real(const std::complex<scalar>& f) { return std::imag(f) == 0.0; } 
    
    double real(double f) { return f; } 
    float real(float f) { return f; } 
    template <typename scalar> inline scalar real(const std::complex<scalar>& f) { return std::real(f); }
    
    
    
    EigenList::~EigenList() {}
    
    /**
     * 
     * Each value is indexed
     * 
     * @author B. Piwowarski <benjamin@bpiwowar.net>
     */
    template <typename scalar> class IndexedValue {
    public:
        /**
         * The rank in the matrix
         */
        size_t newPosition;
        
        /**
         * The new corresponding eigenvalue ( >= original)
         */
        double lambda;
        
        /**
         * The original eigenvalue
         */
        double d;
        
        /**
         * The corresponding z
         */
        scalar z;
        
        /**
         * The status (selection)
         */
        std::bitset<2> status;
        
        /**
         * Original position (-1 if not part of the original matrix)
         */
        long position;
        
        /**
         * Set the selection status
         * 
         * @param value
         */
        void setSelected(bool value) {
            status.set(0, value);
        }
        
        /**
         * Was this eigenvalue removed from the decomposition (by the
         * selection algorithm)
         */
        bool isRemoved() const {
            return status[1];
        }
        
        /**
         * Set the removed value (see {@link #isRemoved()}
         */
        void setRemoved(bool value) {
            status.set(1, value);
        }
        
        /**
         * Was this eigenvalue selected (otherwise it was deflated through
         * rotation or zeroing)
         */
        bool isSelected() const {
            return status[0];
        }
        
        
        IndexedValue() : position(-1) {
        }
        
        IndexedValue(size_t position, double d, scalar z) {
            this->position = position;
            this->d = d;
            this->z = z;
            this->lambda = d;
        }
        
        std::string toString() {
            std::stringstream ss;
            ss << "rank" << newPosition << ", position" << position << ", lambda" << lambda
            << ", d=" << d << ", z=" << z << ", s=" << isSelected() << ", r=" << isRemoved();
            return ss.str();
        }
    };
    
    
    template<typename scalar>   
    class LambdaComparator {
    public:
        LambdaComparator() {}
        bool operator() (const IndexedValue<scalar>* i, const IndexedValue<scalar>* j) { 
            // Special rule to order after the removed parts
            int z = (i->isRemoved() ? 1 : 0) - (j->isRemoved() ? 1 : 0);
            if (z != 0)
                return z;
            
            // We have to invert the order since we want biggest
            // values
            // first - that is, returns -1 if o1 > o2
            return i->lambda > j->lambda;
        }
    };
    
    template<typename scalar> 
    struct DiagonalComparator {
        DiagonalComparator() {}
        bool operator() (const IndexedValue<scalar>* i, const IndexedValue<scalar>* j) { 
            return i->d > j->d;
        } 
    };
    
    
    
    /**
     * A list of eigenvalues
     * 
     * @author B. Piwowarski <benjamin@bpiwowar.net>
     */
    template<typename scalar> class EigenValues : public EigenList {
    public:
        std::vector<IndexedValue<scalar>*>& values;
        std::size_t minRemoved;
        std::size_t rank;
        
        EigenValues(std::vector<IndexedValue<scalar>*>& _values) : values(_values) {
            this->rank = values.size();
            this->minRemoved = values.size();
        }
        
        
        double get(std::size_t index) const {
            return values[index]->lambda;
        }
        
        
        void remove(std::size_t index) {
            if (!values[index]->isRemoved()) {
                minRemoved = std::min(index, minRemoved);
                values[index]->setRemoved(true);
                rank--;
            }
        }
        
        
        std::size_t size() const {
            return values.size();
        }
        
        
        bool isSelected(std::size_t i) const {
            return !values[i]->isRemoved();
        }
        
        
        std::size_t getRank() const {
            return rank;
        }
        
    };
    
    
    
    /**
     * @param v
     */
    template <class Compare, typename scalar> void sortValues(std::vector<IndexedValue<scalar>*>& v, size_t from,
                                                              Compare &compare) {
        
        std::sort(v.begin() + from, v.end(), compare);
    }
    
    inline double norm(double x) { return x*x; }
    inline float norm(float x) { return x*x; }
    template <typename scalar> inline scalar norm(const std::complex<scalar> &z) { return std::norm(z); }
    
    
    /**
     * Compute the value of z
     * 
     * @param v
     * @param M
     * @param lambda0
     * @param i
     * @param vi
     * @param di
     * @param newz
     * @return
     */
    template <typename scalar> scalar computeZ(const std::vector<IndexedValue<scalar>*>& v, int M,
                                               double lambda0, int i, const IndexedValue<scalar>& vi, double di,
                                               bool debug) {
        double normZ = -(di - lambda0);
        
        // lambda_j < di
        for (int j = i + 1; j < M; j++) {
            IndexedValue<scalar> &vj = *v[j];
            normZ *= (di - vj.lambda) / (di - vj.d);
        }
        
        for (int j = 1; j <= i; j++) {
            normZ *= (di - v[j]->lambda) / (di - v[j - 1]->d);
        }
        
        // ensures the norm of zi is the same as newz
        scalar new_z = vi.z * (scalar)( sqrt(normZ) / sqrt(kqp::norm(vi.z)));
        KQP_LOG_DEBUG(logger, "New z" << convert(i) << " = " << convert(vi.z));
        //        newz = kqp::real(vi.z) >= 0 ? sqrt(newz) : -sqrt(newz);
        
        return new_z;
    }
    
    
    
    template<typename scalar> class Rotation {
    public:
        // The rotation matrix [ c, s; -s, c ]
        scalar c, s;
        
        // Which is the singular value column we rotated with
        const IndexedValue<scalar> *vi, *vj;
        
        Rotation(scalar c, scalar s, const IndexedValue<scalar>* _vi, const IndexedValue<scalar>* _vj): vi(_vi), vj(_vj) {
            this->c = c;
            this->s = s;
        }
    };
    
    
    template<typename scalar>
    FastRankOneUpdate<scalar>::FastRankOneUpdate() : gamma(10.) {}

    
    
    template <typename scalar>
    void FastRankOneUpdate<scalar>::update(const Eigen::Matrix<scalar, Eigen::Dynamic, 1> & D, 
                                   double rho, const Eigen::Matrix<scalar, Eigen::Dynamic, 1> & z,
                                   bool computeEigenvectors, const Selector *selector, bool keep,
                                   EvdUpdateResult<scalar> &result,
                                   Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> * Z) {
        
        typedef Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<scalar, Eigen::Dynamic, 1> Vector;
        
        
        // ---
        // --- Deflate the matrix (see. G.W. Steward, p. 176)
        // ---
        
        // The deflated diagonal and corresponding z
        // The matrix we are working on is a (N + 1, N)
        std::size_t N = z.size();
        std::size_t rankD = D.rows();
        
        KQP_LOG_DEBUG(logger, "EVD rank-one update in dimension " << convert(rankD));
        if (rankD > N)
            BOOST_THROW_EXCEPTION(illegal_argument_exception() << errinfo_message("D and z are of compatible dimensions"));
        
        
        // The algorithm assumes that rho > 0, so that we keep track of this
        // here, and modify rho so that it is strictly positive
        bool negativeUpdate = rho < 0;
        if (negativeUpdate)
            rho = -rho;
        
        // The norm of ||D||
        double normD = 0;
        
        // Our store for singular values, D, and z
        
        std::vector<IndexedValue<scalar> > indexedValues(N); // this one is just for memory allocation
        
        typedef std::vector<IndexedValue<scalar>*> ScalarPtrVector;
        ScalarPtrVector v(N);
        
        // Copy the diagonal entries (possibly inserting zeros if needed)
        std::size_t offset = N - rankD;
        
        // i' is the position on the D diagonal
        long iprime = rankD - 1;
        bool foundNonNegative = false;
        bool toSort = false;
        
        for (long i = N; --i >= 0;) {
            double di = 0;
            
            if (!is_real(D(iprime))) 
                BOOST_THROW_EXCEPTION(illegal_argument_exception() << errinfo_message("Diagonal value is not real"));
            
            
            // Check the current D[i', i'] if necessary
            if (iprime >= 0 && offset > 0 && !foundNonNegative)
                foundNonNegative = kqp::real(D(iprime)) >= 0;
            
            long position = 0;
            long index = negativeUpdate ? N - 1 - i : i;
            
            // If i' points on the first non negative
            scalar zpos = 0;
            if (iprime < 0 || (offset > 0 && foundNonNegative)) {
                // di is zero in that part
                // and zi comes from the end of the z vector
                offset--;
                position = z.size() + offset - (N - rankD);
                zpos = z(position);
            } else {
                di = kqp::real(D(iprime));
                di = negativeUpdate ? -di : di;
                position = i - offset;
                zpos = z(position);
                iprime--;
            }
            
            normD += di * di;
            
            // Just checking
            long previousIndex = negativeUpdate ? N - 1 - (i + 1) : i + 1;
            if (i != N - 1 && (negativeUpdate ^ (v[previousIndex]->d > di))) {
                toSort = true;
            }
            
            // If the update is negative, we have to reverse the order since
            // we take the opposite of diagonal entries
            v[index] = &(indexedValues[index] = IndexedValue<scalar>(position, di, ((scalar)sqrt(rho)) * zpos));
        }
        normD = sqrt(normD);
        
        // Sort if needed
        if (toSort) {
            static const DiagonalComparator<scalar> comparator;
            sortValues(v, 0, comparator);
        }
        
        // M is the dimension of the deflated diagonal matrix
        int M = 0;
        
        double tau = gamma * EPSILON;
        double tauM2 = tau * normD;
        double mzNorm = 0;
        
        // The list of rotations
        std::vector<Rotation<scalar> > rotations;
        
        // Deflate the matrix and order the singular values
        IndexedValue<scalar> *last;
        for (int i = 0; i < N; i++) {
            IndexedValue<scalar> &vi = *v[i];
            scalar zi = vi.z;
            
            if (std::abs(zi) <= tauM2) {
            } else if (M > 0 && (last->d - vi.d <= tauM2)) {
                double r = sqrt(kqp::norm(last->z) + kqp::norm(zi));
                rotations.push_back(Rotation<scalar>(last->z / (scalar)r, zi / (scalar)r, last, &vi));
                last->z = r;
                vi.z = 0;
            } else {
                // Else just copy the values
                last = &vi;
                mzNorm += kqp::norm(zi);
                M++;
                vi.setSelected(true);
            }
        }
        
        // Order the array v
        // so that the first M ones are within the
        // deflated matrix and the N-M last are outside
        // and preserving the order
        // last is the last non selected here
        int lastFree = -1;
        if (N != M)
            for (int i = 0; i < N; i++) {
                IndexedValue<scalar> &vi = *v[i];
                if (!vi.isSelected()) {
                    if (lastFree < 0)
                        lastFree = i;
                } else if (lastFree >= 0) {
                    // We have some room here
                    v[i] = v[lastFree];
                    v[lastFree] = &vi;
                    if (lastFree + 1 < i)
                        lastFree++;
                    else
                        lastFree = i;
                }
            }
        
        // ---
        // --- Search for singular values (solving the secular equation)
        // ---
        
        // For the stopping criterion
        double e = gamma * EPSILON * M;
        KQP_LOG_DEBUG(logger, "Computing " << convert(M) << " eigenvalues");
        for (int j = 0; j < M; j++) {
            IndexedValue<scalar> &svj = *v[j];
            double diagj = svj.d;
            
            double interval = (j == 0 ? mzNorm : v[j - 1]->d - diagj) / 2;
            double middle = diagj + interval;
            
            // Stopping criteria from Gu & Eisenstat
            double psi = 0;
            double phi = 0;
            
            // "Searching for singular value between %e and %e (interval %e)", 
            // diagj, diagj + interval * 2, interval);
            
            double nu = -interval;
            double f = -1;
            do {
                // Update nu
                // TODO enhance the root finder by using a better
                // approximation
                double oldnu = nu;
                if (f < 0)
                    nu += interval;
                else
                    nu -= interval;
                
                if (nu == oldnu) {
                    // Stopping since we don't change f anymore
                    break;
                }
                
                // Compute the new phi, psi and f
                psi = phi = 0;
                
                // lambda is between diagj and (diagj1 + diagj)/2
                for (int i = j; i < M; i++) {
                    IndexedValue<scalar> &vi = *v[i];
                    psi += kqp::norm(vi.z) / (vi.d - middle - nu);
                }
                
                for (int i = 0; i < j; i++) {
                    IndexedValue<scalar> &vi = *v[i];
                    phi += kqp::norm(vi.z) / (vi.d - middle - nu);
                }
                
                f = 1 + psi + phi;
                
                interval /= 2;
            } while (std::abs(f) == std::numeric_limits<double>::infinity()
                     || std::abs(f) > (1 + std::abs(psi) + std::abs(phi)) * e);
            
            // Done, store the eigen value
            KQP_LOG_DEBUG(logger, "Eigenvalue: old = " << convert(svj.lambda) << ", computed = " << convert(middle + nu));
            svj.lambda = middle + nu;
            
            // Because of rounding errors, that can happen
            if (svj.lambda < diagj) {
                svj.lambda = diagj;
            } else {
                double max = j == 0 ? mzNorm + diagj : v[j - 1]->d;
                if (svj.lambda > max) {
                    svj.lambda = max;
                }
            }
        }
        
        // }
        
        // ---
        // --- Compute the singular vectors
        // ---
        
        // First, recompute z to match the singular values we have
        
        double lambda0 = v[0]->lambda;
        
        for (int i = 0; i < M; i++) {
            IndexedValue<scalar> vi = *v[i];
            double di = vi.d;
            scalar newz = computeZ(v, M, lambda0, i, vi, di, false);
            
            // Remove z too close to 0
            if (std::abs(newz) < tauM2) {
                v[i]->setSelected(false);
            }
            vi.z = newz;
        }
        
        // --- Let's construct the result ---
        
        // --- First, take the opposite of eigenvalues if
        // --- we are doing a negative update
        if (negativeUpdate)
            for(typename ScalarPtrVector::iterator iv = v.begin(); iv != v.end(); iv++) {
                (*iv)->d = -(*iv)->d;
                (*iv)->lambda = -(*iv)->lambda;
            }
        
        // --- Set eigen values (and the rank)
        
        // Select the eigenvalues if needed
        static const LambdaComparator<scalar> lambdaComparator;
        sortValues(v, 0, lambdaComparator);
        
        size_t rank = v.size();
        if (selector) {
            EigenValues<scalar> list(v);
            selector->selection(list);
            rank = list.rank;
            
            // Reorder if needed
            if (rank < N && (list.minRemoved != rank)) {
                sortValues(v, list.minRemoved, lambdaComparator);
            }
            
        }
        
        // then store them,
        int nbSelected = 0;
        int nbNaN = 0;
        result.mD = Eigen::DiagonalMatrix<scalar, Eigen::Dynamic>(rank);
        for (size_t i = 0; i < rank; i++) {
            v[i]->newPosition = i;
            result.mD.diagonal()(i) = v[i]->lambda;
            if (v[i]->isSelected())
                nbSelected++;
            if (isNaN(v[i]->lambda))
                nbNaN++;
        }
        
        if (nbNaN > 0)
            BOOST_THROW_EXCEPTION(arithmetic_exception() << errinfo_message("We had some eigen value(s) that is/are NaN"));
        
        // --- Compute the eigenvectors
        
        if (computeEigenvectors) {
            // Creates the matrix
            result.mQ.setZero(N, rank);
            Matrix &Q = result.mQ;
            
            // Set the new values
            for (int j = 0; j < rank; j++) {
                IndexedValue<scalar> &vj = *v[j];
                if (!vj.isSelected()) {
                    Q(vj.position, j) = 1;
                } else {
                    // Compute the new vector
                    double columnNorm = 0;
                    int iM = 0;
                    for (int i = 0; i < N && iM < M; i++) {
                        IndexedValue<scalar> &vi = *v[i];
                        if (vi.isSelected()) {
                            double di = vi.d;
                            scalar x = vi.z / (scalar)(di - vj.lambda);
                            columnNorm += kqp::norm(x);
                            Q(vi.position, j) = x;
                            
                            if (isNaN(x)) 
                                BOOST_THROW_EXCEPTION(arithmetic_exception() << errinfo_message("NaN value in matrix Q"));
                            iM++;
                        }
                    }
                    
                    // Normalize
                    Q.col(j).normalize();
                }
            }
            
            // --- Rotate the vectors that need to be rotated
            for (size_t r = 0; r < rotations.size(); r++) {
                Rotation<scalar> &rot = rotations[r];
                size_t i = rot.vi->position;
                size_t j = rot.vj->position;
                // TODO: use Eigen rotation
                
                // Rotation only affect the two rows i and j
                for (int col = 0; col < rank; col++) {
                    scalar x = Q(i,col);
                    scalar y = Q(j,col);
                    Q(i, col) =  x * rot.c - y * rot.s;
                    Q(j, col) = x * rot.s + y * rot.c;
                }
            }
            
            if (!keep && rank < N)
                Q = Q.block(0,0,rank,rank);
            
            if (Z) {
                // TODO: optimise this
                Q = (*Z) * Q;
            }
            
        }
        
        
    }
    
    //explicit instantiation of 
#define RANK_ONE_UPDATE(scalar) template class FastRankOneUpdate<scalar>;
    
    RANK_ONE_UPDATE(double);
    RANK_ONE_UPDATE(float);
    RANK_ONE_UPDATE(std::complex<double>);
    RANK_ONE_UPDATE(std::complex<float>);
}


