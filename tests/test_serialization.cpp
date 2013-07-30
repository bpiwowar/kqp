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

// Test the serialization of a decomposition

#include <sstream>      // std::ostringstream
#include <fstream>


#include <kqp/kqp.hpp>
#include <kqp/serialization.hpp>
#include <kqp/decomposition.hpp>
#include <kqp/feature_matrix/sparse_dense.hpp>
#include <kqp/feature_matrix/dense.hpp>

// include headers that implement a archive in simple text format
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>


using namespace kqp;



namespace Eigen {
    enum {
      ForceNonZeroDiag = 1,
      MakeLowerTriangular = 2,
      MakeUpperTriangular = 4,
      ForceRealDiag = 8
    };
    
/* Initializes both a sparse and dense matrix with same random values,
 * and a ratio of \a density non zero entries.
 * \param flags is a union of ForceNonZeroDiag, MakeLowerTriangular and MakeUpperTriangular
 *        allowing to control the shape of the matrix.
 * \param zeroCoords and nonzeroCoords allows to get the coordinate lists of the non zero,
 *        and zero coefficients respectively.
 */
template<typename Scalar,int Opt2,typename Index> void
initSparse(double density,
           int rows, int cols,
           SparseMatrix<Scalar,Opt2,Index>& sparseMat,
           int flags = 0,
           std::vector<Matrix<Index,2,1> >* zeroCoords = 0,
           std::vector<Matrix<Index,2,1> >* nonzeroCoords = 0)
{
  enum { IsRowMajor = SparseMatrix<Scalar,Opt2,Index>::IsRowMajor };
  sparseMat.setZero();
  //sparseMat.reserve(int(refMat.rows()*refMat.cols()*density));
  sparseMat.reserve(VectorXi::Constant(IsRowMajor ? rows : cols, int((1.5*density)*(IsRowMajor?cols:rows))));
  
  for(Index j=0; j<sparseMat.outerSize(); j++)
  {
    //sparseMat.startVec(j);
    for(Index i=0; i<sparseMat.innerSize(); i++)
    {
      int ai(i), aj(j);
      if(IsRowMajor)
        std::swap(ai,aj);
      Scalar v = (internal::random<double>(0,1) < density) ? internal::random<Scalar>() : Scalar(0);
      if ((flags&ForceNonZeroDiag) && (i==j))
      {
        v = internal::random<Scalar>()*Scalar(3.);
        v = v*v + Scalar(5.);
      }
      if ((flags & MakeLowerTriangular) && aj>ai)
        v = Scalar(0);
      else if ((flags & MakeUpperTriangular) && aj<ai)
        v = Scalar(0);

      if ((flags&ForceRealDiag) && (i==j))
        v = internal::real(v);

      if (v!=Scalar(0))
      {
        //sparseMat.insertBackByOuterInner(j,i) = v;
        sparseMat.insertByOuterInner(j,i) = v;
        if (nonzeroCoords)
          nonzeroCoords->push_back(Matrix<Index,2,1> (ai,aj));
      }
      else if (zeroCoords)
      {
        zeroCoords->push_back(Matrix<Index,2,1> (ai,aj));
      }
    }
  }
  //sparseMat.finalize();
}
} // eigen namespace


template<typename T>
void saveAndLoad(const T &a, T &b) {
    std::ostringstream oss;
    {
        boost::archive::text_oarchive ar(oss);
        ar & a;
    }

    // std::cout << oss.str() << std::endl;
    
    // --- Load
    {
        std::istringstream iss(oss.str());
        boost::archive::text_iarchive ar(iss);
        ar & b;
    }
    
}

template<typename Scalar> int test_dense_matrix() 
{
    KQP_SCALAR_TYPEDEFS(Scalar);
    ScalarMatrix m1(ScalarMatrix::Random(5, 5));
    ScalarMatrix m2;
    
    saveAndLoad(m1, m2);
    
    double delta = (m1 - m2).squaredNorm();
    std::cerr << "delta(ScalarMatrix)=" << delta << std::endl;
    
    return delta > 0 ? 1 : 0;
}

int error_of(const std::string &what, double delta)
{
    std::cerr << "delta(" << what << ")=" << delta << std::endl; 
    return delta > 0 ? 1 : 0;
}

template<typename Scalar> int test_alt_matrix() {
    KQP_SCALAR_TYPEDEFS(Scalar);
    int errors;
    
    ScalarAltMatrix m1(ScalarMatrix(ScalarMatrix::Random(5, 5)));
    ScalarAltMatrix m2;
    saveAndLoad(m1, m2);
    
    errors += error_of("AltMatrix[1]", (ScalarMatrix(m1) - ScalarMatrix(m2)).squaredNorm());

    m1 = IdentityScalarMatrix(5,5);
    saveAndLoad(m1, m2);
    errors += error_of("AltMatrix[2]", (ScalarMatrix(m1) - ScalarMatrix(m2)).squaredNorm());
    
    return errors;
}
template<typename Scalar> int test_alt_vector() {
    KQP_SCALAR_TYPEDEFS(Scalar);
    int errors;
    
    RealAltVector m1(ScalarVector(ScalarVector::Random(5)));
    RealAltVector m2;
    saveAndLoad(m1, m2);
    
    errors += error_of("AltVector[1]", (ScalarVector(m1) - ScalarVector(m2)).squaredNorm());

    int a = -42;
    int b;
    saveAndLoad(a,b);
    
    m1 = ConstantRealVector(5,1,1.2);
    saveAndLoad(m1, m2);
    errors += error_of("AltVector[2]", (ScalarVector(m1) - ScalarVector(m2)).squaredNorm());
    
    return errors;
}


template<typename Scalar> int test_decomposition() {
    // --- Create

    KQP_SCALAR_TYPEDEFS(Scalar);
    int dim = 50;
    int dim2 = 10;

    boost::shared_ptr<DenseSpace<Scalar>> fs(new DenseSpace<Scalar>(dim));
    
    ScalarMatrix m = ScalarMatrix::Random(dim, dim2);

    auto mX = fs->newMatrix(m);
    ScalarAltMatrix mY(ScalarMatrix(ScalarMatrix::Random(dim, dim2)));
    RealAltVector mD(ScalarVector::Random(dim2));
    
    kqp::Decomposition<Scalar> d1(fs, mX, mY, mD, false);
    kqp::Decomposition<Scalar> d2;
    
    saveAndLoad(d1, d2);
    
    double deltaX = (d1.mX->template as<Dense<Scalar>>().toDense() - d2.mX->template as<Dense<Scalar>>().toDense()).squaredNorm();
    std::cerr << "[dense] delta(X)=" << deltaX << std::endl;

    double deltaY = (ScalarMatrix(d1.mY) - ScalarMatrix(d2.mY)).squaredNorm();
    std::cerr << "[dense] delta(Y)=" << deltaY << std::endl;

    double deltaD = (ScalarVector(d1.mD) - ScalarVector(d2.mD)).squaredNorm();
    std::cerr << "[dense] delta(D)=" << deltaD << std::endl;
    
    return (deltaX > 0 || deltaY > 0 || deltaD > 0) ? 1 : 0;
}

template<typename Scalar> int test_sparse_dense_decomposition() {
    // --- Create

    KQP_SCALAR_TYPEDEFS(Scalar);
    int dim = 1000;
    int dim2 = 10;

    boost::shared_ptr<SparseDenseSpace<Scalar>> fs(new SparseDenseSpace<Scalar>(dim));
    
    Eigen::SparseMatrix<double> m(dim, dim2);
    initSparse(0.2, dim, dim2, m);
    auto mX = SparseDense<Scalar>::create(m);
    
    ScalarAltMatrix mY(ScalarMatrix(ScalarMatrix::Random(dim, dim2)));
    RealAltVector mD(ScalarVector::Random(dim2));
    
    kqp::Decomposition<Scalar> d1(fs, mX, mY, mD, false);
    kqp::Decomposition<Scalar> d2;
    
    saveAndLoad(d1, d2);
    
    double deltaX = (d1.mX->template as<SparseDense<Scalar>>().toDense() - d2.mX->template as<SparseDense<Scalar>>().toDense()).squaredNorm();
    std::cerr << "[sparse-dense] delta(X)=" << deltaX << std::endl;

    double deltaY = (ScalarMatrix(d1.mY) - ScalarMatrix(d2.mY)).squaredNorm();
    std::cerr << "[sparse-dense] delta(Y)=" << deltaY << std::endl;

    double deltaD = (ScalarVector(d1.mD) - ScalarVector(d2.mD)).squaredNorm();
    std::cerr << "[sparse-dense] delta(D)=" << deltaD << std::endl;
    
    return (deltaX > 0 || deltaY > 0 || deltaD > 0) ? 1 : 0;
}



int main(int, char**) {
    int error = 0;
    error += test_dense_matrix<double>();
    error += test_alt_matrix<double>();
    error += test_alt_vector<double>();
    
    error += test_decomposition<double>();
    error += test_sparse_dense_decomposition<double>();
    if (error > 0)
        KQP_THROW_EXCEPTION_F(kqp::assertion_exception, "%d error(s) in serialization", %error);
}