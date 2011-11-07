#include <algorithm>
#include <bitset>
#include <vector>
#include <sstream>
#include <cmath>
#include <limits.h>
#include <exception>

#include "kqp.h"
#include "kernel_evd.h"
#include "evd_update.h"

using namespace kqp;

EigenList::~EigenList() {}

/**
 * 
 * Each value is indexed
 * 
 * @author B. Piwowarski <benjamin@bpiwowar.net>
 */
class IndexedValue {
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
    double z;
    
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
    
    IndexedValue(size_t position, double d, double z) {
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


struct LambdaComparator {
    bool operator() (const IndexedValue* i, const IndexedValue* j) { 
        // Special rule to order after the removed parts
        int z = (i->isRemoved() ? 1 : 0) - (j->isRemoved() ? 1 : 0);
        if (z != 0)
            return z;
        
        // We have to invert the order since we want biggest
        // values
        // first - that is, returns -1 if o1 > o2
        return i->lambda > j->lambda;
    }
} LAMBDA_COMPARATOR;

struct DiagonalComparator {
    bool operator() (const IndexedValue* i, const IndexedValue* j) { 
        return i->d > j->d;
    } 
} DIAGONAL_COMPARATOR;



/**
 * A list of eigenvalues
 * 
 * @author B. Piwowarski <benjamin@bpiwowar.net>
 */
class EigenValues : public EigenList {
public:
    std::vector<IndexedValue*>& values;
    std::size_t minRemoved;
    std::size_t rank;
    
    EigenValues(std::vector<IndexedValue*>& _values) : values(_values) {
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
template <class Compare> void sortValues(std::vector<IndexedValue*>& v, size_t from,
                                         Compare &compare) {
    
    std::sort(v.begin() + from, v.end(), compare);
}

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
double computeZ(const std::vector<IndexedValue*>& v, int M,
                double lambda0, int i, const IndexedValue& vi, double di,
                bool debug) {
    double newz = -(di - lambda0);
    
    // lambda_j < di
    for (int j = i + 1; j < M; j++) {
        IndexedValue &vj = *v[j];
        newz *= (di - vj.lambda) / (di - vj.d);
    }
    
    for (int j = 1; j <= i; j++) {
        newz *= (di - v[j]->lambda) / (di - v[j - 1]->d);
    }
    
    newz = vi.z >= 0 ? sqrt(newz) : -sqrt(newz);
    
    return newz;
}


inline double sqr(double x) { return x*x; }

class Rotation {
public:
    // The rotation matrix [ c, s; -s, c ]
    double c, s;
    
    // Which is the singular value column we rotated with
    const IndexedValue *vi, *vj;
    
    Rotation(double c, double s, const IndexedValue* _vi, const IndexedValue* _vj): vi(_vi), vj(_vj) {
        this->c = c;
        this->s = s;
    }
};


FastRankOneUpdate::FastRankOneUpdate() : gamma(10.) {}


void FastRankOneUpdate::rankOneUpdate(const boost::shared_ptr<Eigen::MatrixXd>& Z,
                                        const Eigen::VectorXd& D, double rho, const Eigen::VectorXd& z,
                                        bool computeEigenvectors, const Selector *selector, bool keep,
                                      EvdUpdateResult &result) {
    // ---
    // --- Deflate the matrix (see. G.W. Steward, p. 176)
    // ---
    
    // The deflated diagonal and corresponding z
    // The matrix we are working on is a (N + 1, N)
    std::size_t N = z.size();
    std::size_t rankD = D.cols();
    
    if (rankD > N)
        // "D and z  are not compatible in broken arrow SVD"
        throw new std::exception();
    
    
    // The algorithm assumes that rho > 0, so that we keep track of this
    // here, and modify rho so that it is strictly positive
    bool negativeUpdate = rho < 0;
    if (negativeUpdate)
        rho = -rho;
    
    // The norm of ||D||
    double normD = 0;
    
    // Our store for singular values, D, and z
    
    std::vector<IndexedValue> indexedValues(N); // this one is just for memory allocation
    
    std::vector<IndexedValue*> v(N);
    
    // Copy the diagonal entries (possibly inserting zeros if needed)
    std::size_t offset = N - rankD;
    
    // i' is the position on the D diagonal
    long iprime = rankD - 1;
    bool foundNonNegative = false;
    bool toSort = false;
    
    for (long i = N; --i >= 0;) {
        double di = 0;
        
        // Check the current D[i', i'] if necessary
        if (iprime >= 0 && offset > 0 && !foundNonNegative)
            foundNonNegative = D(iprime, iprime) >= 0;
        
        long position = 0;
        long index = negativeUpdate ? N - 1 - i : i;
        
        // If i' points on the first non negative
        double zpos = 0;
        if (iprime < 0 || (offset > 0 && foundNonNegative)) {
            // di is zero in that part
            // and zi comes from the end of the z vector
            offset--;
            position = z.size() + offset - (N - rankD);
            zpos = z(position);
        } else {
            di = D(iprime, iprime);
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
        v[index] = &(indexedValues[index] = IndexedValue(position, di, sqrt(rho) * zpos));
    }
    normD = sqrt(normD);
    
    // Sort if needed
    if (toSort) {
        sortValues(v, 0, DIAGONAL_COMPARATOR);
    }
    
    // M is the dimension of the deflated diagonal matrix
    int M = 0;
    
    double tau = gamma * EPSILON;
    double tauM2 = tau * normD;
    double mzNorm = 0;
    
    // The list of rotations
    std::vector<Rotation> rotations;
    
    // Deflate the matrix and order the singular values
    IndexedValue *last;
    for (int i = 0; i < N; i++) {
        IndexedValue &vi = *v[i];
        double zi = vi.z;
        
        if (std::abs(zi) <= tauM2) {
        } else if (M > 0 && (last->d - vi.d <= tauM2)) {
            double r = sqrt(sqr(last->z) + sqr(zi));
            rotations.push_back(Rotation(last->z / r, zi / r, last, &vi));
            last->z = r;
            vi.z = 0;
        } else {
            // Else just copy the values
            last = &vi;
            mzNorm += zi * zi;
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
            IndexedValue &vi = *v[i];
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
    
    for (int j = 0; j < M; j++) {
        IndexedValue &svj = *v[j];
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
                IndexedValue &vi = *v[i];
                psi += sqr(vi.z) / (vi.d - middle - nu);
            }
            
            for (int i = 0; i < j; i++) {
                IndexedValue &vi = *v[i];
                phi += sqr(vi.z) / (vi.d - middle - nu);
            }
            
            f = 1 + psi + phi;
            
            interval /= 2;
        } while (std::abs(f) == std::numeric_limits<double>::infinity()
                 || std::abs(f) > (1 + std::abs(psi) + std::abs(phi)) * e);
        
        // Done, store the eigen value
        // logger.info("LAPACK value = %g, computed value = %g", svj.lambda,
        // middle + nu);
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
        IndexedValue vi = *v[i];
        double di = vi.d;
        double newz = computeZ(v, M, lambda0, i, vi, di, false);
                
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
        for(std::vector<IndexedValue*>::iterator iv = v.begin(); iv != v.end(); iv++) {
            (*iv)->d = -(*iv)->d;
            (*iv)->lambda = -(*iv)->lambda;
        }
    
    // --- Set eigen values (and the rank)
    
    // Select the eigenvalues if needed
    sortValues(v, 0, LAMBDA_COMPARATOR);
    
    size_t rank = v.size();
    if (selector) {
        EigenValues list(v);
        selector->selection(list);
        rank = list.rank;
        
        // Reorder if needed
        if (rank < N && (list.minRemoved != rank)) {
            sortValues(v, list.minRemoved, LAMBDA_COMPARATOR);
        }
        
    }
    
    // then store them,
    int nbSelected = 0;
    int nbNaN = 0;
    result.mD = Eigen::DiagonalMatrix<double,Eigen::Dynamic>(rank);
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
        Eigen::MatrixXd &Q = (result.mQ = Eigen::MatrixXd(N, rank));
        
        // Set the new values
        for (int j = 0; j < rank; j++) {
            IndexedValue &vj = *v[j];
            if (!vj.isSelected()) {
                Q(vj.position, j) = 1;
            } else {
                // Compute the new vector
                double columnNorm = 0;
                int iM = 0;
                for (int i = 0; i < N && iM < M; i++) {
                    IndexedValue &vi = *v[i];
                    if (vi.isSelected()) {
                        double di = vi.d;
                        double x = vi.z / (di - vj.lambda);
                        columnNorm += x * x;
                        Q(vi.position, j) = x;
                        
                        if (isNaN(x)) 
                            BOOST_THROW_EXCEPTION(arithmetic_exception());
                        iM++;
                    }
                }
                
                // Normalize
                Q.col(j).normalize();
            }
        }
        
        // --- Rotate the vectors that need to be rotated
        for (size_t r = 0; r < rotations.size(); r++) {
            Rotation &rot = rotations[r];
            size_t i = rot.vi->position;
            size_t j = rot.vj->position;
            // TODO: use Eigen rotation
            
            // Rotation only affect the two rows i and j
            for (int col = 0; col < rank; col++) {
                double x = Q(i,col);
                double y = Q(j,col);
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
