#include <algorithm>
#include <bitset>
#include <vector>
#include <sstream>
#include <cmath>
#include <limits.h>
#include <exception>
#include <complex>

#include <kqp/kqp.hpp>
#include <kqp/evd_update.hpp>

DEFINE_LOGGER(logger, "kqp.evd-update");

namespace kqp {
    
    // Useful Scalar dependant functions
    
    inline bool is_real(double) { return true; } 
    inline bool is_real(float) { return true; } 
    template <typename Scalar> inline bool is_real(const std::complex<Scalar>& f) { return std::imag(f) == 0.0; } 
    
    double real(double f) { return f; } 
    float real(float f) { return f; } 
    template <typename Scalar> inline Scalar real(const std::complex<Scalar>& f) { return std::real(f); }
    
    
    
    template<typename Scalar> EigenList<Scalar>::~EigenList() {}
    
    /**
     * 
     * Each value is indexed
     * 
     * @author B. Piwowarski <benjamin@bpiwowar.net>
     */
    template <typename Scalar> class IndexedValue {
    public:
        typedef typename Eigen::NumTraits<Scalar>::Real Real;

        /**
         * The rank in the matrix
         */
        size_t newPosition;
        
        /**
         * The new corresponding eigenvalue ( >= original)
         */
        Real lambda;
        
        /**
         * The original eigenvalue
         */
        Real d;
        
        /**
         * The corresponding z
         */
        Scalar z;
        
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
        
        IndexedValue(size_t position, Real d, Scalar z) {
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
    
    
    template<typename Scalar>   
    class LambdaComparator {
    public:
        LambdaComparator() {}
        bool operator() (const IndexedValue<Scalar>* i, const IndexedValue<Scalar>* j) { 
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
    
    template<typename Scalar> 
    struct DiagonalComparator {
        DiagonalComparator() {}
        bool operator() (const IndexedValue<Scalar>* i, const IndexedValue<Scalar>* j) { 
            return i->d > j->d;
        } 
    };
    
    
    
    /**
     * A list of eigenvalues
     * 
     * @author B. Piwowarski <benjamin@bpiwowar.net>
     */
    template<typename Scalar> class EigenValues : public EigenList<typename Eigen::NumTraits<Scalar>::Real> {
    public:
        typedef typename Eigen::NumTraits<Scalar>::Real Real;
        std::vector<IndexedValue<Scalar>*>& values;
        Index minRemoved;
        Index rank;
        
        EigenValues(std::vector<IndexedValue<Scalar>*>& _values) : values(_values) {
            this->rank = values.size();
            this->minRemoved = values.size();
        }
        
        
        Real get(Index index) const {
            check(index);
            return values[index]->lambda;
        }
        
        
        void remove(Index index) {
            check(index);
            if (!values[index]->isRemoved()) {
                minRemoved = std::min(index, minRemoved);
                values[index]->setRemoved(true);
                rank--;
            }
        }
        
        
        Index size() const {
            return values.size();
        }
        
        
        bool isSelected(std::size_t i) const {
            check(i);
            return !values[i]->isRemoved();
        }
        
        
        Index getRank() const {
            return rank;
        }
        
    private:
#ifdef NDEBUG
        inline void check(std::size_t ) const {}
#else
        inline void check(std::size_t i) const {
            assert(i < (std::size_t)size());
        }
#endif
    };
    
    
    
    /**
     * @param v
     */
    template <class Compare, typename Scalar> void sortValues(std::vector<IndexedValue<Scalar>*>& v, size_t from,
                                                              Compare &compare) {
        
        std::sort(v.begin() + from, v.end(), compare);
    }
    
    inline double norm(double x) { return x*x; }
    inline float norm(float x) { return x*x; }
    template <typename Scalar> inline Scalar norm(const std::complex<Scalar> &z) { return std::norm(z); }
    
    
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
    template <typename Scalar> Scalar computeZ(const std::vector<IndexedValue<Scalar>*>& v, int M,
                                               double lambda0, int i, const IndexedValue<Scalar>& vi, double di) {
        double normZ = -(di - lambda0);
        
        // lambda_j < di
        for (int j = i + 1; j < M; j++) {
            IndexedValue<Scalar> &vj = *v[j];
            normZ *= (di - vj.lambda) / (di - vj.d);
        }
        
        for (int j = 1; j <= i; j++) {
            normZ *= (di - v[j]->lambda) / (di - v[j - 1]->d);
        }
        
        // ensures the norm of zi is the same as newz
        Scalar new_z = vi.z * (Scalar)( sqrt(normZ) / sqrt(kqp::norm(vi.z)));
        KQP_LOG_DEBUG(logger, "New z" << convert(i) << " = " << convert(new_z) << " / old was " << convert(vi.z));
        //        newz = kqp::real(vi.z) >= 0 ? sqrt(newz) : -sqrt(newz);
        
        return new_z;
    }
    
    
    
    template<typename Scalar> class Rotation {
    public:
        // The rotation matrix [ c, s; -s, c ]
        Scalar c, s;
        
        // Which is the singular value column we rotated with
        const IndexedValue<Scalar> *vi, *vj;
        
        Rotation(Scalar c, Scalar s, const IndexedValue<Scalar>* _vi, const IndexedValue<Scalar>* _vj): vi(_vi), vj(_vj) {
            this->c = c;
            this->s = s;
        }
    };
    
    
    template<typename Scalar>
    FastRankOneUpdate<Scalar>::FastRankOneUpdate() : gamma(10.) {}

    
    
    template <typename Scalar>
    void FastRankOneUpdate<Scalar>::update(const Eigen::Matrix<Real,Dynamic,1> & D, 
                                   double rho, const Eigen::Matrix<Scalar,Dynamic,1> & z,
                                           bool computeEigenvectors, const Selector<typename FastRankOneUpdate<Scalar>::Real> *selector, bool keep,
                                   EvdUpdateResult<Scalar> &result,
                                   Eigen::Matrix<Scalar,Dynamic,Dynamic> * Z) {
        
        typedef Eigen::Matrix<Scalar,Dynamic,Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar,Dynamic,1> Vector;
        
        
        // ---
        // --- Deflate the matrix (see. G.W. Steward, p. 176)
        // ---
        
        // The deflated diagonal and corresponding z
        // The matrix we are working on is a (N + 1, N)
        Index N = z.size();
        Index rankD = D.rows();
        
        KQP_LOG_DEBUG_F(logger, "EVD rank-one update in dimension %d", %std::max(rankD,N));
        if (rankD > N)
            KQP_THROW_EXCEPTION_F(illegal_argument_exception, "D and z are of incompatible dimensions (%d and %d)", %rankD %N);
        
        
        // The algorithm assumes that rho > 0, so that we keep track of this
        // here, and modify rho so that it is strictly positive
        bool negativeUpdate = rho < 0;
        if (negativeUpdate)
            rho = -rho;
        
        // The norm of ||D||
        double normD = 0;
        
        // Our store for singular values, D, and z
        
        std::vector<IndexedValue<Scalar> > indexedValues(N); // this one is just for memory allocation
        
        typedef std::vector<IndexedValue<Scalar>*> ScalarPtrVector;
        ScalarPtrVector v(N);
        
        // Copy the diagonal entries (possibly inserting zeros if needed)
        Index offset = N - rankD;
        
        // i' is the position on the D diagonal
        long iprime = rankD - 1;
        bool foundNonNegative = false;
        bool toSort = false;
        
        for (Index i = N; --i >= 0;) {
            Real di = 0;
            
            if (iprime >= 0 && !is_real(D(iprime))) 
                BOOST_THROW_EXCEPTION(illegal_argument_exception() << errinfo_message("Diagonal value is not real"));
            
            
            // Check the current D[i', i'] if necessary
            if (iprime >= 0 && offset > 0 && !foundNonNegative)
                foundNonNegative = kqp::real(D(iprime)) >= 0;
            
            size_t position = 0;
            size_t index = negativeUpdate ? N - 1 - i : i;
            
            // If i' points on the first non negative
            Scalar zpos = 0;
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
            v[index] = &(indexedValues[index] = IndexedValue<Scalar>(position, di, ((Scalar)sqrt(rho)) * zpos));
        }
        normD = sqrt(normD);
        
        // Sort if needed
        if (toSort) {
            static const DiagonalComparator<Scalar> comparator;
            sortValues(v, 0, comparator);
        }
        
        // M is the dimension of the deflated diagonal matrix
        int M = 0;
        
        double tau = gamma * EPSILON;
        double tauM2 = tau * normD;
        double mzNorm = 0;
        
        // The list of rotations
        std::vector<Rotation<Scalar> > rotations;
        
        // Deflate the matrix and order the singular values
        IndexedValue<Scalar> *last = 0;
        for (int i = 0; i < N; i++) {
            IndexedValue<Scalar> &vi = *v[i];
            Scalar zi = vi.z;
            
            if (std::abs(zi) <= tauM2) {
            } else if (M > 0 && (last->d - vi.d <= tauM2)) {
                double r = sqrt(kqp::norm(last->z) + kqp::norm(zi));
                rotations.push_back(Rotation<Scalar>(last->z / (Scalar)r, zi / (Scalar)r, last, &vi));
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
                IndexedValue<Scalar> &vi = *v[i];
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
            IndexedValue<Scalar> &svj = *v[j];
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
                    IndexedValue<Scalar> &vi = *v[i];
                    psi += kqp::norm(vi.z) / (vi.d - middle - nu);
                }
                
                for (int i = 0; i < j; i++) {
                    IndexedValue<Scalar> &vi = *v[i];
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
        
        double lambda0 = v.empty() ? 0 : v[0]->lambda;
        
        for (int i = 0; i < M; i++) {
            IndexedValue<Scalar> vi = *v[i];
            double di = vi.d;
            Scalar newz = computeZ(v, M, lambda0, i, vi, di);
            
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
        static const LambdaComparator<Scalar> lambdaComparator;
        std::sort(v.begin(), v.end(), lambdaComparator);
               
        Index rank = v.size();
        if (selector) {
            EigenValues<Scalar> list(v);
            selector->selection(list);
            rank = list.rank;
            
            // Reorder if needed
            if (rank < N && (list.minRemoved != rank)) {
                std::sort(v.begin() + list.minRemoved, v.end(), lambdaComparator);
            }
            
        }
        
        // then store them,
        int nbSelected = 0;
        int nbNaN = 0;
        KQP_LOG_DEBUG_F(logger, "Final rank is %d", %rank);
        result.mD = RealVector(rank);
        for (Index i = 0; i < rank; i++) {
            v[i]->newPosition = i;
            result.mD(i) = v[i]->lambda;
            if (v[i]->isSelected())
                nbSelected++;
            if (isNaN(v[i]->lambda))
                nbNaN++;
        }
        
        if (nbNaN > 0)
            KQP_THROW_EXCEPTION_F(arithmetic_exception, "We had %d eigen value(s) that is/are NaN", %nbNaN);
        
        // --- Compute the eigenvectors
        
        if (computeEigenvectors || Z) {
            // Creates the matrix
            result.mQ.setZero(N, rank);
            Matrix &Q = result.mQ;
            
            // Set the new values: work eigenvector by eigenvector (indexed by j)
            for (int j = 0; j < rank; j++) {
                IndexedValue<Scalar> &vj = *v[j];
                if (!vj.isSelected()) {
                    Q(vj.position, j) = 1;
                } else {
                    // Compute the new vector
                    double columnNorm = 0;
                    int iM = 0;
                    for (Index i = 0; i < N && iM < M; i++) {
                        IndexedValue<Scalar> &vi = *v[i];
                        if (vi.isSelected()) {
                            double di = vi.d;
                            Scalar x = vi.z / (Scalar)(di - vj.lambda);
                            columnNorm += kqp::norm(x);
                            Q(vi.position, j) = x;
                            
                            if (isNaN(x)) 
                                KQP_THROW_EXCEPTION(arithmetic_exception, "NaN value in matrix Q");
                            iM++;
                        }
                    }
                    
                    // Normalize
                    Q.col(j) /= std::sqrt(columnNorm);
                }
            }
            
            // --- Rotate the vectors that need to be rotated
            for (size_t r = 0; r < rotations.size(); r++) {
                Rotation<Scalar> &rot = rotations[r];
                size_t i = rot.vi->position;
                size_t j = rot.vj->position;
                // TODO: use Eigen rotation
                
                // Rotation only affect the two rows i and j
                for (int col = 0; col < rank; col++) {
                    Scalar x = Q(i,col);
                    Scalar y = Q(j,col);
                    Q(i, col) =  x * rot.c - y * rot.s;
                    Q(j, col) = x * rot.s + y * rot.c;
                }
            }
            
            if (!keep && rank < N)
                Q = Q.block(0,0,rank,rank);
            
            if (Z) {
                if (Z->cols() < rank) {
                    Index z_cols = Z->cols();
                    Index z_rows = Z->rows();
                    
                    Index diff = rank - z_cols;
                    Z->conservativeResize(Z->rows() + diff, Z->cols() + diff);
                    
                    Z->bottomLeftCorner(diff, z_cols).setConstant(0);
                    Z->bottomRightCorner(diff, diff).setIdentity(diff, diff);
                    Z->topRightCorner(z_rows, diff).setConstant(0);
                }
                *Z = ((*Z) * Q).eval();

            }
            
        }
        
        
    }
    
    //explicit instantiation of 
#define RANK_ONE_UPDATE(Scalar) template class FastRankOneUpdate<Scalar>; template class EigenList<Scalar>;
    
    RANK_ONE_UPDATE(double);
    RANK_ONE_UPDATE(float);
    RANK_ONE_UPDATE(std::complex<double>);
    RANK_ONE_UPDATE(std::complex<float>);
}


