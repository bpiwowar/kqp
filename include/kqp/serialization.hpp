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

#ifndef _KQP_SERIALIZATION_H
#define _KQP_SERIALIZATION_H

#include <boost/serialization/level.hpp>
// #include <boost/serialization/version.hpp>
#include <boost/serialization/split_free.hpp>
//     serialization::split_free(ar, t, file_version); 


#include <boost/preprocessor/comma.hpp> 
#include <kqp/decomposition.hpp>
#include <kqp/feature_matrix/dense.hpp>
#include <kqp/feature_matrix/sparse_dense.hpp>


namespace boost
{
    // namespace serialization 
    // {
    // template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    // struct implementation_level<Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>
    // {
    //     typedef mpl::integral_c_tag tag;
    //     typedef mpl::int_<object_serializable> type;
    //     BOOST_STATIC_CONSTANT(int, value = level_type::object_serializable);
    // };
    // }
    // 
    using namespace kqp;



//! Direct loader (when there is no default constructor)
template<typename Archive, typename T>
struct Loader {
    static inline T load(Archive & ar) {
        T t;
        ar & t;
        return t;
    }
};
    
// -- Eigen matrix
template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
inline void serialize(
    Archive & ar, 
    Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & t, 
    const unsigned int /*file_version*/
) 
{
    Index rows = t.rows(), cols = t.cols();
    ar & rows;
    ar & cols;
    
    if( rows != t.rows() || cols != t.cols() )
        t.resize( rows, cols );

    for(Index i=0; i<t.size(); i++)
        ar & t.data()[i];
}

namespace serialization 
{
template<typename Scalar, typename Matrix>
struct implementation_level<Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<Scalar>,Matrix>>
{
    typedef mpl::integral_c_tag tag;
    typedef mpl::int_<object_serializable> type;
    BOOST_STATIC_CONSTANT(
        int,
        value = level_type::object_serializable
    );
};

template<typename Scalar, typename Matrix>
struct tracking_level<Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<Scalar>,Matrix>>
{
    typedef mpl::integral_c_tag tag;
    typedef mpl::int_<track_never> type;
    BOOST_STATIC_CONSTANT(
        int, 
        value = tracking_type::track_never
    );
};
}


template<class Archive, typename Scalar, typename Matrix>
inline void save(
    Archive & ar, 
    const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<Scalar>, Matrix> & t, 
    const unsigned int /*file_version*/
) 
{
    size_t rows = t.rows(), cols = t.cols();
    ar & rows;
    ar & cols;
    
    Scalar value = t.coeff(0,0);
    ar & value;
}


template<typename Archive, typename Scalar, typename Matrix>
struct Loader<Archive, Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<Scalar>, Matrix>> {
    static inline Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<Scalar>, Matrix> load(Archive &ar) 
    {
        size_t rows, cols;
        Scalar value;

        ar & rows;
        ar & cols;
        ar & value;
        
        return Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<Scalar>, Matrix>(rows, cols, value);
    }
};

template<class Archive, typename Scalar, typename Matrix>
inline void serialize(
    Archive & ar, 
    Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<Scalar>, Matrix> & t, 
    const unsigned int file_version
) 
{
    serialization::split_free(ar, t, file_version); 
}




// -- Alt matrix
    
template<class Archive, typename T1, typename T2>
inline void save(
    Archive & ar, 
    const kqp::AltMatrix<T1, T2>& t, 
    const unsigned int /*file_version*/
) 
{
    bool isT1 = t.isT1();
    ar & isT1;
    
    if (isT1) {
        const T1 & t1 = t.t1();
        ar & t1;
    } else {
        const T2 & t2 = t.t2();
        ar & t2;
    }
}



    
template<class Archive, typename T1, typename T2>
inline void load(
    Archive & ar, 
    kqp::AltMatrix<T1, T2>& t, 
    const unsigned int /*file_version*/
) 
{
    bool isT1;
    ar & isT1;
    
    if (isT1)
    {
        // std::cerr << "LOADING " << KQP_DEMANGLE(T1) << "/" << KQP_DEMANGLE(t)  << std::endl;
        t = Loader<Archive, T1>::load(ar);        
    } 
    else 
    {
        // std::cerr << "LOADING " << KQP_DEMANGLE(T2) << "/" << KQP_DEMANGLE(t) << std::endl;
        t = Loader<Archive, T2>::load(ar);        
    }
    
}


    
template<class Archive, typename T1, typename T2>
inline void serialize(
    Archive & ar, 
    kqp::AltMatrix<T1, T2>& t, 
    const unsigned int file_version
) 
{
    serialization::split_free(ar, t, file_version); 
}





// --- Feature Matrix

template<class Archive, typename Scalar>
inline void save(
    Archive & ar, 
    const boost::shared_ptr<FeatureMatrixBase<Scalar>> &m, 
    const unsigned int /*file_version*/
) 
{
    std::string type;

    if (dynamic_cast<kqp::Dense<Scalar>*>(m.get()) != 0) 
    {
        type = "dense";
        ar << type;
        ar & dynamic_cast<kqp::Dense<Scalar>&>(*m).getMatrix();
    }
    else if (dynamic_cast<kqp::SparseDense<Scalar>*>(m.get()) != 0) 
    {
        type = "sparse-dense";
        ar << type;
        ar & dynamic_cast<kqp::SparseDense<Scalar>&>(*m);
    }
    else 
    {
        KQP_THROW_EXCEPTION_F(kqp::not_implemented_exception, "Cannot serialize feature matrix of type %s", %KQP_DEMANGLE(*m.get()));
    }
}

template<class Archive, typename Scalar>
inline void load(
    Archive & ar, 
    boost::shared_ptr<FeatureMatrixBase<Scalar>> &t, 
    const unsigned int /*file_version*/
) 
{
    std::string type;
    ar & type;
    
    if (type == "dense") 
    {
        typename kqp::Dense<Scalar>::ScalarMatrix m;
        ar & m;
        t = kqp::Dense<Scalar>::create(std::move(m));
    } 
    else if (type == "sparse-dense") 
    {
        boost::shared_ptr<SparseDense<Scalar>> p(new kqp::SparseDense<Scalar>());
        ar & *p;
        t = p;
    } 
    else 
    {
        KQP_THROW_EXCEPTION_F(kqp::not_implemented_exception, "Cannot unserialize feature matrix of type %s", %type);
    }
}

template<class Archive, typename Scalar>
inline void serialize(
    Archive & ar, 
    boost::shared_ptr<FeatureMatrixBase<Scalar>> &t, 
    const unsigned int file_version
) 
{
    serialization::split_free(ar, t, file_version); 
}

} // ns(boost)



#endif
