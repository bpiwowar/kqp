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
#ifndef __KQP_SPARSE_DENSE_FEATURE_MATRIX_H__
#define __KQP_SPARSE_DENSE_FEATURE_MATRIX_H__

#include <boost/lexical_cast.hpp>
#include <boost/function_output_iterator.hpp>

#include <kqp/subset.hpp>
#include <kqp/feature_matrix.hpp>
#include <Eigen/Sparse>


namespace kqp {
    
    /**
     * @brief A feature matrix where vectors are in a dense subspace (in the canonical basis).
     *
     * This class makes the hypothesis that vectors have only a few non null components (compared to the dimensionality of the space), 
     * and that those components are mostly the same.
     *
     * In practice, the matrix is a map from a row index to a vector (along with a count of the number of zeros),
     * where each vector has a size less or equal to the number of columns of the sparse matrix.
     * @ingroup FeatureMatrix
     */
    template <typename Scalar> 
    class SparseDense : public FeatureMatrixBase<Scalar> {
    public:
        KQP_SCALAR_TYPEDEFS(Scalar);
        typedef SparseDense<Scalar> Self;

        typedef std::map<Index,Index> RowMap;

        virtual ~SparseDense() {}
        
        SparseDense() :  m_dimension(0) {}
        SparseDense(Index dimension) :  m_dimension(dimension) {}
        SparseDense(const Self &other) :  m_dimension(other.m_dimension), m_map(other.m_map), m_matrix(other.m_matrix), m_gramMatrix(other.m_gramMatrix) {}

        //! Creates from a matrix
        static FMatrix create(const ScalarMatrix &m) {
            return FMatrix(new Self(m));
        }
        
        //! Creates from a matrix (from a column major sparse matrix)
        static FMatrix create(const Eigen::SparseMatrix<Scalar, Eigen::ColMajor> &m, double threshold = EPSILON) {
            return FMatrix(new Self(m, threshold));
        }

        //! Creates from a matrix (from a row major sparse matrix)
        static FMatrix create(const Eigen::SparseMatrix<Scalar, Eigen::RowMajor> &m, double threshold = EPSILON) {
            return FMatrix(new Self(m, threshold));
        }
        
        
#ifndef SWIG
        SparseDense(Index dimension, RowMap &&map, ScalarMatrix &&matrix)
            : m_dimension(dimension), m_map(std::move(map)), m_matrix(std::move(matrix)) {
            
        }
#endif

        //! Creates from a sparse matrix
        SparseDense(const Eigen::SparseMatrix<Scalar, Eigen::ColMajor> &mat, double threshold = EPSILON) : m_dimension(mat.rows()) {
            // --- Compute which rows we need
            
            // Computing the column norms
            RealVector norms(mat.cols());
            norms.setZero();
            for (Index k=0; k<mat.cols(); ++k)  { // Loop on cols
                // FIXME: use norm() when Eigen fixed
                // norms[k] = mat.innerVector(k).norm();
                for (typename Eigen::SparseMatrix<Scalar, Eigen::ColMajor>::InnerIterator it(mat,k); it; ++it) { // Loop on rows 
                    norms[k] += Eigen::internal::abs2(it.value());
                }
            }
            
            norms = norms.cwiseSqrt();
            
            // Computing selected rows map
            Index rows = 0;
            for (int k=0; k<mat.outerSize(); ++k) { // Loop on cols
                for (typename Eigen::SparseMatrix<Scalar, Eigen::ColMajor>::InnerIterator it(mat,k); it; ++it) { // Loop on rows
                    if (std::abs(it.value()) / norms[it.col()] > threshold) {
                        if (m_map.find(it.row()) == m_map.end()) {
                          m_map[it.row()] = rows;
                          rows += 1;                            
                        }
                    }
                }
            }
            
            // --- Copying
            assert(rows == (Index)m_map.size());
            m_matrix.resize(rows, mat.cols());
            m_matrix.setZero();
            
            if (rows > 0)
                for (int k=0; k<mat.outerSize(); ++k) { // Loop on cols
                    for (typename Eigen::SparseMatrix<Scalar, Eigen::ColMajor>::InnerIterator it(mat,k); it; ++it) { // Loop on rows
                        if (std::abs(it.value()) / norms[it.col()] > threshold) 
                            m_matrix(m_map[it.row()], it.col()) = it.value();
                    }
                }
            
        }

        
        //! Creates from a sparse matrix
        SparseDense(const Eigen::SparseMatrix<Scalar, Eigen::RowMajor> &mat, double threshold = EPSILON) : m_dimension(mat.rows()) {
            // --- Compute which rows we need
            
                      
            // Computing the column norms
            RealVector norms(mat.cols());
            norms.setZero();
            for (int k=0; k<mat.outerSize(); ++k)  // Loop on rows
                for (typename Eigen::SparseMatrix<Scalar, Eigen::RowMajor>::InnerIterator it(mat,k); it; ++it) // Loop on cols
                    norms[it.col()] += Eigen::internal::abs2(it.value());

            // Computing selected rows
            norms = norms.cwiseSqrt();
            Index rows = 0;
            for (int k=0; k<mat.outerSize(); ++k) {
                bool any = false;
                for (typename Eigen::SparseMatrix<Scalar, Eigen::RowMajor>::InnerIterator it(mat,k); it && !any; ++it) {
                    any |= std::abs(it.value()) / norms[it.col()] > threshold;
                }
                if (any) {
                    m_map[k] = rows;
                    rows += 1;
                }
            }

            // --- Copying
            m_matrix.resize(rows, mat.cols());
            m_matrix.setZero();
            
            for(auto i = m_map.begin(); i != m_map.end(); i++) {
                for (typename Eigen::SparseMatrix<Scalar, Eigen::RowMajor>::InnerIterator it(mat,i->first); it; ++it) 
                    m_matrix(i->second, it.col()) = it.value();

            }
            
        }
        
        /**
         * @brief Construct from a dense matrix
         *
         * Discard rows where entries are neglectable (see the class documentation).
         *
         * @param mat The matrix to copy
         * @param threshold The threshold for a value to be neglectable
         */
        SparseDense(const ScalarMatrix &mat, double threshold = Eigen::NumTraits<Scalar>::epsilon()) : m_dimension(mat.rows()) {
            // Compute which rows we need
            Matrix<Real, 1, Dynamic> norms = mat.cwiseAbs2().colwise().sum();
            
            size_t rows = 0;
            for(Index i = 0; i < mat.rows(); i++) {
                if ((std::abs(mat.row(i).array()) / norms.array() > threshold).any()) {
                    m_map[i] = rows;
                    rows += 1;
                }
            }
            
            // Copy the selected rows
            m_matrix.resize(rows, mat.cols());
            if ((size_t)mat.rows() == m_map.size())
                m_matrix = mat;
            else 
                for(auto i = m_map.begin(); i != m_map.end(); i++) {
                    m_matrix.row(i->second) = mat.row(i->first);
                }
            
        }
        
        
        //! Converts to a dense matrix (mostly used for debug)
        ScalarMatrix toDense() const {
            ScalarMatrix mat(m_dimension, size());
            mat.setZero();
            
            for(auto i = m_map.begin(); i != m_map.end(); i++)
                mat.row(i->first) = m_matrix.row(i->second);
            
            return mat;
        }
        
        //! Get the dense dimension
        Index denseDimension() const {
            return m_matrix.rows();
        }
        
        /** @brief Cleanup the near zero entries
         *
         * An entry of a vector is considered as zero if its ratio to the maximum magnitude component of the vectors
         * is above a given threshold. If a full row is considered as below zero, it is removed.
         *
         * @param threshold The threshold above which a value is considered as zero
         */
        void cleanup(Real threshold = EPSILON) {
            Matrix<Real, 1, Dynamic> norms = m_matrix.cwiseAbs2().colwise().sum();

            std::vector<bool> selected(m_matrix.rows(), true);
            for(auto i = m_map.begin(); i != m_map.end(); ) { 
                auto j = i;
                i++;
                if ((std::abs(m_matrix.row(j->second).array()) / norms.array() <= threshold).all()) {
                    selected[j->second] = false;
                    m_map.erase(j);
                }
            }
            
            Index newSize = 0;
            std::vector<Index> ix(m_matrix.rows(), 0);
            for(Index i = 0; i < m_matrix.rows(); i++) {
                ix[i] = newSize;
                if (selected[i]) newSize++;
            }
            for(auto i = m_map.begin(); i != m_map.end(); i++) 
                i->second = ix[i->second];
            
            
            select_rows(selected, m_matrix, m_matrix);            
            assert(m_matrix.rows() == newSize);
        }
        
        // --- Base methods 
        virtual Index size() const { 
            return m_matrix.cols();
        }
        
        Index dimension() const {
            return m_dimension;
        }

        
#ifndef SWIG
        struct Insert {
            RowMap &m_map;
            void operator()(const RowMap::const_reference &x) const {
                size_t s = m_map.size();
                m_map[x.first] = s;
            }
        };
        
        struct KeyComparator {
            bool operator()(const RowMap::const_reference &a, const RowMap::const_reference b) {
                return a.first < b.first;
            }
        };
#endif
        
        void add(const FMatrixBase &_other, const std::vector<bool> *which = NULL) override {
            const Self &other = dynamic_cast<const Self&>(_other);
            
            if (m_dimension != other.m_dimension)
                KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Cannot add vectors of different sizes (%d vs %d)", %m_dimension %other.m_dimension);
            
            // Compute which vectors to add
            std::vector<Index> ix;
            Index toAdd = 0;
            if (which) {
                for(size_t i = 0; i < which->size(); i++)
                    if ((*which)[i]) ix.push_back(i);
                toAdd = ix.size();
            } else toAdd = other.size();
            
            if (toAdd == 0) return;
            
            Index oldRows = m_map.size();
            std::set_difference(other.m_map.begin(), other.m_map.end(), m_map.begin(), m_map.end(), 
                                boost::make_function_output_iterator(Insert({m_map})), KeyComparator());
            
            Index newRows = m_map.size() - oldRows;
            
            // Add 
            Index offset = m_matrix.cols();
            m_matrix.conservativeResize(m_map.size(), m_matrix.cols() + toAdd);
            m_matrix.topRightCorner(oldRows, toAdd).setZero();
            m_matrix.bottomRows(newRows).setZero();
            
            for(auto i = other.m_map.begin(); i != other.m_map.end(); i++) {
                Index otherRow = i->second;
                Index selfRow = m_map[i->first];
                
                if (which) {
                    for(size_t k = 0; k < ix.size(); k++)
                        m_matrix(selfRow, offset + k) = other.m_matrix(otherRow, ix[k]);
                } else
                    m_matrix.row(selfRow).tail(toAdd) = other.m_matrix.row(otherRow);
            }
                        
        }
        
        
        const ScalarMatrix &gramMatrix() const {
            if (size() == m_gramMatrix.rows()) return m_gramMatrix;
            
            // We lose space here, could be used otherwise???
            Index current = m_gramMatrix.rows();
            if (current < size()) 
                m_gramMatrix.conservativeResize(size(), size());
            
            Index tofill = size() - current;
            
            // Compute the remaining inner products
            m_gramMatrix.bottomRightCorner(tofill, tofill).noalias() = m_matrix.rightCols(tofill).adjoint() * m_matrix.rightCols(tofill);
            m_gramMatrix.topRightCorner(current, tofill).noalias() = m_matrix.leftCols(current).adjoint() * m_matrix.rightCols(tofill);
            m_gramMatrix.bottomLeftCorner(tofill, current) = m_gramMatrix.topRightCorner(current, tofill).adjoint().eval();
            
            return m_gramMatrix;
        }
        
        
        //! Computes the inner product with another matrix
        template<class DerivedMatrix>
        void inner(const Self &other, DerivedMatrix &result) const {
            if (result.rows() != this->size() || result.cols() != other.size())
              result.resize(this->size(), other.size());
            result.setZero();
            
            struct Collector {
                const ScalarMatrix &mat1, &mat2;
                const RowMap &map1, &map2;
                DerivedMatrix &mat;
                
                void operator()(const RowMap::const_reference & x) const {
                    Index ix1 = this->map1.find(x.first)->second;
                    Index ix2 = this->map2.find(x.first)->second;
                    this->mat += this->mat1.row(ix1).adjoint() * this->mat2.row(ix2);
                }
            } collector({m_matrix, other.m_matrix, m_map, other.m_map, result});

            std::set_intersection(m_map.begin(), m_map.end(), other.m_map.begin(), other.m_map.end(), 
                                  boost::make_function_output_iterator(collector), KeyComparator());
            
        }
        
        
        // Computes alpha * X * A + beta * Y * B (X = *this)
        FMatrixBasePtr linearCombination(const ScalarAltMatrix &mA, Scalar alpha, const Self *mY, const ScalarAltMatrix *mB, Scalar beta) const {
            // Simple case: we don't have to add another matrix
            if (!mY) 
                return FMatrixBasePtr(new Self(m_dimension, RowMap(m_map), alpha * m_matrix * mA));
            
            // Add the keys
            RowMap newMap;
            std::set_union(mY->m_map.begin(), mY->m_map.end(), m_map.begin(), m_map.end(), 
                           boost::make_function_output_iterator(Insert({newMap})),
                           KeyComparator());


            // Perform the linear combination
            ScalarMatrix mat(newMap.size(), mA.cols());
            mat.setZero();
            
            for(auto i = m_map.begin(), end = m_map.end(); i != end; i++) 
                mat.row(newMap[i->first]) = alpha * m_matrix.row(i->second) * mA;

            for(auto i = mY->m_map.begin(), end = mY->m_map.end(); i != end; i++) 
                mat.row(newMap[i->first]) += beta * mY->m_matrix.row(i->second) * *mB;
            
            // Move and cleanup before returning
            FMatrixBasePtr sdMat(new Self(m_dimension, std::move(newMap), std::move(mat)));
            dynamic_cast<Self&>(*sdMat).cleanup(EPSILON);
            return sdMat;
        }
        
        
        
        FMatrixBasePtr subset(const std::vector<bool>::const_iterator &begin, const std::vector<bool>::const_iterator &end) const override {
            boost::shared_ptr<Self> dest(new Self());
            select_columns(begin, end, m_matrix, dest->m_matrix);
            
            dest->m_dimension = m_dimension;
            dest->m_map = m_map;
            
            return dest;
        }
        
        virtual FMatrixBasePtr copy() const override {
            return FMatrixBasePtr(new Self(*this));
        }

        Self &operator=(const Self &other)  {
            m_dimension = other.m_dimension;
            m_matrix = other.m_matrix;
            m_map = other.m_map;
            m_gramMatrix = other.m_gramMatrix;
            return *this;
        }


        virtual FMatrixBase &operator=(const FMatrixBase &_other) override {
          return *this = dynamic_cast<const Self&>(_other);
        }
        
        //! Get the sparse to dense map
        const std::map<Index, Index> &map() const {
            return m_map;
        }
        
        //! Get the dense matrix
        const ScalarMatrix &getMatrix() const {
            return m_matrix;
        }

    private:
        //! Dimension of the space
        Index m_dimension;
                
        //! A map from row to row
        std::map<Index, Index> m_map;

        //! The dense matrix
        ScalarMatrix m_matrix;

        //! Cache of the gram matrix
        mutable ScalarMatrix m_gramMatrix;

    };
    
    
    template<typename Scalar>
    class SparseDenseSpace : public SpaceBase<Scalar> {
    public:  
        KQP_SCALAR_TYPEDEFS(Scalar);
#ifndef SWIG
        using SpaceBase<Scalar>::k;
#endif        
        static FSpace create(Index dimension) { return FSpace(new SparseDenseSpace(dimension)); }
        
        SparseDenseSpace(Index dimension) : m_dimension(dimension) {}
        SparseDenseSpace() : m_dimension(0) {}
        
        inline static const SparseDense<Scalar>& cast(const FeatureMatrixBase<Scalar> &mX) { return dynamic_cast<const SparseDense<Scalar> &>(mX); }
        
        
        Index dimension() const override { return m_dimension; }
        void dimension(Index dimension) { m_dimension = dimension; }

        virtual FSpacePtr copy() const override { return FSpacePtr(new SparseDenseSpace(m_dimension));  }

        virtual FMatrixBasePtr newMatrix() const override {
            return FMatrixBasePtr(new SparseDense<Scalar>(m_dimension));
        }
        virtual FMatrixBasePtr newMatrix(const FMatrixBase &mX) const override {
            return FMatrixBasePtr(new SparseDense<Scalar>(cast(mX)));            
        }
        
        virtual bool _canLinearlyCombine() const override {
            return true;
        }
        
        const ScalarMatrix &k(const FeatureMatrixBase<Scalar> &mX) const override {
            return cast(mX).gramMatrix();
        }
        
        virtual ScalarMatrix k(const FeatureMatrixBase<Scalar> &mX1, const ScalarAltMatrix &mY1, const RealAltVector &mD1,
                               const FeatureMatrixBase<Scalar> &mX2, const ScalarAltMatrix &mY2, const RealAltVector &mD2) const override {        
            ScalarMatrix inner;
            cast(mX1).inner(cast(mX2), inner);
            return mD1.asDiagonal() * mY1.adjoint() * inner * mY2 * mD2.asDiagonal();
        };
 
        virtual FMatrixBasePtr linearCombination(const FeatureMatrixBase<Scalar> &mX, const ScalarAltMatrix &mA, Scalar alpha, 
                                          const FeatureMatrixBase<Scalar> *mY, const ScalarAltMatrix *mB, Scalar beta) const override {
            return cast(mX).linearCombination(mA, alpha, dynamic_cast<const SparseDense<Scalar> *>(mY), mB, beta);
        }
        
        static const std::string &name() { static std::string NAME("sparse-dense"); return NAME; }

        virtual void load(const pugi::xml_node &node) override {
            m_dimension = boost::lexical_cast<Index>(node.attribute("dimension").value());
        }

        virtual void save(pugi::xml_node &node) const override {
            pugi::xml_node self = node.append_child(name().c_str());
            self.append_attribute("dimension") = boost::lexical_cast<std::string>(m_dimension).c_str();
        }

    private:
        Index m_dimension;
        
    };
    
    
    
# // Extern templates
#ifndef SWIG
# define KQP_SCALAR_GEN(scalar) \
  extern template class SparseDense<scalar>; \
  extern template class SparseDenseSpace<scalar>;
# include <kqp/for_all_scalar_gen.h.inc>
#endif    
} // end namespace kqp

#endif