#include <iostream>
#include <ctime>
#include <deque>

#include <boost/lexical_cast.hpp>

#include <kqp/kqp.hpp>
#include <kqp/cleanup.hpp>

#include <kqp/kernel_evd/dense_direct.hpp>
#include <kqp/kernel_evd/accumulator.hpp>
#include <kqp/kernel_evd/incremental.hpp>

DEFINE_LOGGER(logger,  "kqp.benchmark.kernel-evd");

namespace kqp {
    
    
    struct KernelEVDBenchmark {
        
        // Seed
        long seed = 0;
        
        // Space dimension
        Index dimension = 100;
        Index updates = 1000;
        
        // Range for the number of pre-images at each update
        Index min_preimages = 1;
        Index max_preimages = 1;
        
        // Range for the number of vectors at each update
        Index min_lc = 1;
        Index max_lc = 1;
        
        // Use linear combination
        bool useLinearCombination = true;
        
        // Scalar
        std::string scalarName = "double";
        
        // ---   Kernel-EVD Algorithm
        
        std::string kevdName = "direct";

        // --- Cleaning up
        
        std::pair<float,float> preImageRatios;
        Index maxRank = dimension;
        
        template<typename Builder> 
        int run(Builder &builder) {
            typedef typename Builder::FTraits::FMatrix FMatrix;
            KQP_FMATRIX_TYPES(FMatrix);
            
            
            // --- Computing Kernel EVD
            
            KQP_LOG_INFO_F(logger, "Computing kernel EVD with builder %s", % KQP_DEMANGLE(builder));
            
            std::clock_t c_start = std::clock();
            
            std::srand(seed);
            for(int i = 0; i < updates; i++) {
                
                Real alpha = Eigen::internal::abs(Eigen::internal::random_impl<Real>::run()) + 1e-3;
                
                int k = (int)std::abs(Eigen::internal::random_impl<double>::run() * (double)(max_preimages-min_preimages)) + min_preimages;
                int p = (int)std::abs(Eigen::internal::random_impl<double>::run() * (double)(max_lc-min_lc)) + min_lc;
                KQP_LOG_DEBUG(logger, boost::format("Pre-images (%dx%d) and linear combination (%dx%d)") % dimension % k % k % p);
                
                // Generate a number of pre-images
                ScalarMatrix m = ScalarMatrix::Random(dimension, k);
                
                // Generate the linear combination matrix
                ScalarMatrix mA = ScalarMatrix::Random(k, p);
                
                
                builder.add(alpha, FMatrix(m), mA);
            }        
            
            Decomposition<FMatrix> result = builder.getDecomposition();
            std::clock_t c_end = std::clock();
            std::cout << "kevd\t" << 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC << std::endl;
            
            KQP_LOG_INFO_F(logger, "Decomposition result: %d pre-images and rank %d", %result.mX.size() %result.mY.cols());
            
            // --- Cleaning up
            
            KQP_LOG_INFO(logger, "Cleaning up");
            c_start = std::clock();
            StandardCleaner<FMatrix> cleaner;
            boost::shared_ptr<RankSelector<Real, true>> rankSelector(new RankSelector<Real,true>(maxRank));
            
            cleaner.setSelector(rankSelector);
            cleaner.setPreImagesPerRank(preImageRatios.first, preImageRatios.second);
            cleaner.setUseLinearCombination(useLinearCombination);
            cleaner.cleanup(result);
            
            c_end = std::clock();
            std::cout << "cleaning\t" << 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC << std::endl;
            
            if (!result.orthonormal) {
                KQP_LOG_INFO(logger, "Re-orthonormalizing");
                c_start = std::clock();
                kqp::orthonormalize(result.mX, result.mY, result.mD);

                c_end = std::clock();
                std::cout << "orthonormalizing\t" << 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC << std::endl;
            } else {
                std::cout << "orthonormalizing\t" << 0 << std::endl;
            }
            


            // --- Computing the error
            KQP_LOG_INFO(logger, "Computing the error");
            std::srand(seed);
            
            double orthogononal_error = 0;            
            ScalarMatrix sumWWT(result.mY.cols(), result.mY.cols());
            sumWWT.setZero();
            
            ScalarMatrix mYTXT = result.mY.transpose() * result.mX.get_matrix().transpose();
            
            for(int i = 0; i < updates; i++) {
                
                Real alpha = Eigen::internal::abs(Eigen::internal::random_impl<Real>::run()) + 1e-3;
                
                int k = (int)std::abs(Eigen::internal::random_impl<double>::run() * (double)(max_preimages-min_preimages)) + min_preimages;
                int p = (int)std::abs(Eigen::internal::random_impl<double>::run() * (double)(max_lc-min_lc)) + min_lc;
                KQP_LOG_DEBUG(logger, boost::format("Pre-images (%dx%d) and linear combination (%dx%d)") % dimension % k % k % p);
                
                // Generate the pre-images
                FMatrix mU = FMatrix(ScalarMatrix::Random(dimension, k));
                
                // Generate the linear combination matrix
                ScalarMatrix mA = ScalarMatrix::Random(k, p);
                
                // Project
                ScalarMatrix mW = mYTXT * mU.get_matrix() * mA;
                sumWWT.template selfadjointView<Eigen::Lower>().rankUpdate(mW, alpha);
                orthogononal_error += Eigen::internal::abs(alpha) * (mA.transpose() * mU.inner() * mA - mW.transpose() * mW).squaredNorm();
            }   
            
            std::cout << "o_error\t" << orthogononal_error << std::endl;
            std::cout << "s_error\t" << (ScalarMatrix(sumWWT.template selfadjointView<Eigen::Lower>()) - ScalarMatrix(result.mD.asDiagonal())).squaredNorm() << std::endl;

            std::cout << "pre_images\t" << result.mX.size() << std::endl;
            std::cout << "rank\t" << result.mY.cols() << std::endl;
            return 0;
        }
        
        template<typename Scalar>
        int select_kevd() {
            typedef typename Eigen::NumTraits<Scalar>::Real Real;
            
            if (kevdName == "direct") {
                DenseDirectBuilder<Scalar> builder(dimension);
                return this->run(builder);
            }

            if (kevdName == "incremental") {
                IncrementalKernelEVD<DenseMatrix<Scalar>> builder;
                boost::shared_ptr<RankSelector<Real, true>> rankSelector(new RankSelector<Real,true>(maxRank));
                builder.setSelector(rankSelector);
                builder.setPreImagesPerRank(preImageRatios.first, preImageRatios.second);
                return this->run(builder);
            }

            if (kevdName == "accumulator") {
                if (useLinearCombination) {
                    AccumulatorKernelEVD<DenseMatrix<Scalar>,true> builder;
                    return this->run(builder);
                }
                
                AccumulatorKernelEVD<DenseMatrix<Scalar>,false> builder;
                return this->run(builder);
            }
            

            KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Unknown kernel-evd builder type [%s]", %kevdName);
                
        }
        
        int process(std::deque<std::string> &args) {
            preImageRatios.first = preImageRatios.second = std::numeric_limits<float>::infinity();
            
            // Read the arguments
            while (args.size() > 0) {
                if (args[0] == "--no-lc") {
                    useLinearCombination = false;
                    args.pop_front();
                }

                else if (args[0] == "--seed" && args.size() >= 2) {
                    args.pop_front();
                    seed = std::atol(args[0].c_str());
                    args.pop_front();
                } 
                
                else if (args[0] == "--kevd" && args.size() >= 2) {
                    args.pop_front();
                    kevdName = args[0];
                    args.pop_front();
                }

                else if (args[0] == "--dimension" && args.size() >= 2) {
                    args.pop_front();
                    dimension = boost::lexical_cast<Index>(args[0]);
                    args.pop_front();
                }
                
                else if (args[0] == "--updates" && args.size() >= 2) {
                    args.pop_front();
                    updates = boost::lexical_cast<Index>(args[0]);
                    args.pop_front();
                }

                else if (args[0] == "--pre-image-ratios" && args.size() >= 3) {
                    args.pop_front();
                    preImageRatios.first = boost::lexical_cast<float>(args[0]);
                    preImageRatios.second = boost::lexical_cast<float>(args[1]);
                    args.pop_front();
                    args.pop_front();
                }

                else if (args[0] == "--max-rank" && args.size() >= 2) {
                    args.pop_front();
                    maxRank = boost::lexical_cast<Index>(args[0]);
                    args.pop_front();
                }
                else break;
            }
            
            if (args.size() > 0) 
                KQP_THROW_EXCEPTION_F(illegal_argument_exception, "There are %d unprocessed command line arguments, starting with [%s]", %args.size() %args[0]);
            
            
            // Select the right builder
            if (scalarName == "double") return this->select_kevd<double>();
//            if (scalarName == "float") return this->select_kevd<float>();
//            if (scalarName == "double-complex") return this->select_kevd<std::complex<double>>();
//            if (scalarName == "float-complex") return this->select_kevd<std::complex<float>>();
            
            KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Unknown scalar type [%s]", %scalarName);
        }
        
        
        
        
    };
    
    int bm_kernel_evd(std::deque<std::string> &args) {
        return KernelEVDBenchmark().process(args);
    }
}