#include <iostream>
#include <ctime>
#include <deque>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/algorithm/string.hpp>

#include <kqp/kqp.hpp>
#include <kqp/cleanup.hpp>

#include <kqp/kernel_evd/dense_direct.hpp>
#include <kqp/kernel_evd/accumulator.hpp>
#include <kqp/kernel_evd/incremental.hpp>
#include <kqp/kernel_evd/divide_and_conquer.hpp>

DEFINE_LOGGER(logger,  "kqp.benchmark.kernel-evd");

namespace kqp {
    
    
    struct KernelEVDBenchmark {
        
        
        
        // Id prefix
        std::string id;
        
        // Seed
        long seed = 0;
        
        // Space dimension
        Index dimension = 100;
        Index updates = 1000;
        
        // --- Base vector
        
        // Number of generated vectors
        Index nbVectors;
        
        // Noise ratio for generated vector components
        float noise = 1e-4;
        
        // Binomial for picking a given rank
        
        
        // --- Settings for the generation
        
        // Range for the number of pre-images at each update
        Index min_preimages = 1;
        Index max_preimages = 1;
        
        // Range for the number of vectors at each update
        Index min_lc = 1;
        Index max_lc = 1;
        
        
        
        
        
        // --- 
        
        struct BuilderConfiguratorBase {
            std::pair<float,float> preImageRatios;
            Index targetRank = std::numeric_limits<Index>::max();            
            Index targetPreImageRatio = std::numeric_limits<float>::infinity();
            bool useLinearCombination = true;
            
            virtual void print(std::string prefix = "") {
                std::cout << prefix << "rank\t" << targetRank << std::endl;
                std::cout << prefix << "pre_images\t" << targetPreImageRatio << std::endl;
                std::cout << prefix << "builder\t" << this->getName() << std::endl;
                std::cout << prefix << "lc\t" << useLinearCombination;
            }
            
            virtual std::string getName() const = 0;
            virtual bool processOption(std::deque<std::string> &options, std::deque<std::string> &args) {
                if (options[0] == "rank" && args.size() >= 2) {
                    args.pop_front();
                    targetRank = boost::lexical_cast<Index>(args[0]);
                    args.pop_front();
                    return true;
                }
                
                if (args[0] == "pre-images" && args.size() >= 3) {
                    args.pop_front();
                    targetPreImageRatio = boost::lexical_cast<float>(args[0]);
                    args.pop_front();
                    return true;
                }
                if (args[0] == "no-lc") {
                    useLinearCombination = false;
                    args.pop_front();
                    return true;
                } 

                return false;
            }
            
            virtual int run(const KernelEVDBenchmark &) = 0;
        };
        
        template<typename _Scalar>
        struct BuilderConfigurator : public BuilderConfiguratorBase {
            typedef DenseMatrix<_Scalar> FMatrix;
            KQP_SCALAR_TYPEDEFS(Scalar);
            
            virtual KernelEVD<Scalar>  *getBuilder(const KernelEVDBenchmark &) = 0;
            
            int run(const KernelEVDBenchmark &bm) {
                boost::scoped_ptr<KernelEVD<Scalar> > builder(this->getBuilder(bm));
                
                std::clock_t total_time = 0;
                
                // --- Computing Kernel EVD
                
                KQP_LOG_INFO_F(logger, "Computing kernel EVD with builder %s", % KQP_DEMANGLE(*builder));
                
                std::clock_t c_start = std::clock();
                
                std::srand(bm.seed);
                for(int i = 0; i < bm.updates; i++) {
                    
                    Real alpha = Eigen::internal::abs(Eigen::internal::random_impl<Real>::run()) + 1e-3;
                    
                    int k = (int)std::abs(Eigen::internal::random_impl<double>::run() * (double)(bm.max_preimages-bm.min_preimages)) + bm.min_preimages;
                    int p = (int)std::abs(Eigen::internal::random_impl<double>::run() * (double)(bm.max_lc-bm.min_lc)) + bm.min_lc;
                    KQP_LOG_DEBUG(logger, boost::format("Pre-images (%dx%d) and linear combination (%dx%d)") % bm.dimension % k % k % p);
                    
                    // Generate a number of pre-images
                    ScalarMatrix m = ScalarMatrix::Random(bm.dimension, k);
                    
                    // Generate the linear combination matrix
                    ScalarMatrix mA = ScalarMatrix::Random(k, p);
                    
                    
                    builder->add(alpha, FMatrix(m), mA);
                }        
                
                Decomposition<Scalar> result = builder->getDecomposition();
                std::clock_t c_end = std::clock();
                total_time += c_end-c_start;
                std::cout << "kevd\t" << 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC << std::endl;
                KQP_LOG_INFO_F(logger, "Decomposition result: %d pre-images and rank %d", %result.mX.size() %result.mY.cols());
                
                // --- Cleaning up
                
                KQP_LOG_INFO(logger, "Cleaning up");
                c_start = std::clock();
                StandardCleaner<FMatrix> cleaner;
                
                boost::shared_ptr<RankSelector<Real, true>> rankSelector(new RankSelector<Real,true>(this->targetRank));
                boost::shared_ptr<MinimumSelector<Real>> minSelector(new MinimumSelector<Real>());
                boost::shared_ptr<ChainSelector<Real>> selector(new ChainSelector<Real>());
                selector->add(minSelector);
                selector->add(rankSelector);
                cleaner.setSelector(rankSelector);
                
                cleaner.setPreImagesPerRank(this->targetPreImageRatio, this->targetPreImageRatio);
                cleaner.setUseLinearCombination(this->useLinearCombination);
                cleaner.cleanup(result);
                
                c_end = std::clock();
                total_time += c_end-c_start;
                std::cout << "cleaning\t" << 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC << std::endl;
                
                if (!result.orthonormal) {
                    KQP_LOG_INFO(logger, "Re-orthonormalizing");
                    c_start = std::clock();
                    kqp::orthonormalize(result.mX, result.mY, result.mD);
                    
                    c_end = std::clock();
                    total_time += c_end-c_start;
                    std::cout << "orthonormalizing\t" << 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC << std::endl;
                } else {
                    std::cout << "orthonormalizing\t" << 0 << std::endl;
                }
                
                std::cout << "time\t" << 1000.0 * (total_time) / CLOCKS_PER_SEC << std::endl;
                
                // --- Computing the error
                KQP_LOG_INFO(logger, "Computing the error");
                std::srand(bm.seed);
                
                double orthogononal_error = 0;            
                ScalarMatrix sumWWT(result.mY.cols(), result.mY.cols());
                sumWWT.setZero();
                
                ScalarMatrix mYTXT = result.mY.transpose() * result.mX.get_matrix().transpose();
                
                for(int i = 0; i < bm.updates; i++) {
                    
                    Real alpha = Eigen::internal::abs(Eigen::internal::random_impl<Real>::run()) + 1e-3;
                    
                    int k = (int)std::abs(Eigen::internal::random_impl<double>::run() * (double)(bm.max_preimages-bm.min_preimages)) + bm.min_preimages;
                    int p = (int)std::abs(Eigen::internal::random_impl<double>::run() * (double)(bm.max_lc-bm.min_lc)) + bm.min_lc;
                    KQP_LOG_DEBUG(logger, boost::format("Pre-images (%dx%d) and linear combination (%dx%d)") % bm.dimension % k % k % p);
                    
                    // Generate a number of pre-images
                    ScalarMatrix mU = ScalarMatrix::Random(bm.dimension, k);
                    
                    // Generate the linear combination matrix
                    ScalarMatrix mA = ScalarMatrix::Random(k, p);
                    
                    // Project
                    ScalarMatrix mW = mYTXT * mU * mA;
                    sumWWT.template selfadjointView<Eigen::Lower>().rankUpdate(mW, alpha);
                    orthogononal_error += Eigen::internal::abs(alpha) * (mA.transpose() * mU.transpose() * mU * mA - mW.transpose() * mW).squaredNorm();
                }   
                
                std::cout << "o_error\t" << orthogononal_error << std::endl;
                std::cout << "s_error\t" << (ScalarMatrix(sumWWT.template selfadjointView<Eigen::Lower>()) - ScalarMatrix(result.mD.asDiagonal())).squaredNorm() << std::endl;
                
                std::cout << "pre_images\t" << result.mX.size() << std::endl;
                std::cout << "rank\t" << result.mY.cols() << std::endl;
                return 0;
                
            }
            
        };
        
        
        template<typename Scalar>
        struct DirectConfigurator : public BuilderConfigurator<Scalar> {
            virtual KernelEVD<DenseMatrix<Scalar>> *getBuilder(const KernelEVDBenchmark &bm) override {
                return new DenseDirectBuilder<Scalar>(bm.dimension);
            };
            virtual std::string getName() const { return "direct"; }
        };
        
        template<typename Scalar> 
        struct AccumulatorConfigurator : public BuilderConfigurator<Scalar> {
            
            virtual KernelEVD<DenseMatrix<Scalar>> *getBuilder(const KernelEVDBenchmark &) override {
                if (this->useLinearCombination) 
                    return new AccumulatorKernelEVD<DenseMatrix<Scalar>,true>();
                
                return new AccumulatorKernelEVD<DenseMatrix<Scalar>,false>();
            }
            
            virtual std::string getName() const {
                return "accumulator";
            }
        };
  
        
        template<typename Scalar> 
        struct IncrementalConfigurator : public BuilderConfigurator<Scalar> {
            typedef typename Eigen::NumTraits<Scalar>::Real Real;
            float maxPreImageRatio = std::numeric_limits<float>::infinity();
            Index maxRank = -1;
            
            virtual void print(std::string prefix = "") override {
                std::cout << prefix << "max_pre_images\t" << maxPreImageRatio << std::endl;
                std::cout << prefix << "max_rank\t" << maxRank << std::endl;
            }
            
            virtual bool processOption(std::deque<std::string> &options, std::deque<std::string> &args) override {
                if (options[0] == "max-rank" && args.size() >= 2) {
                    args.pop_front();
                    maxRank = boost::lexical_cast<Index>(args[0]);
                    args.pop_front();
                    return true;
                }
                
                if (args[0] == "max-pre-images" && args.size() >= 2) {
                    args.pop_front();
                    maxPreImageRatio = boost::lexical_cast<float>(args[0]);
                    args.pop_front();
                    return true;
                }
                
                return false;            
            }
        
            virtual std::string getName() const { return "incremental"; }
            
            virtual KernelEVD<DenseMatrix<Scalar>> *getBuilder(const KernelEVDBenchmark &) override {
                // Construct the rank selector
                boost::shared_ptr<RankSelector<Real, true>> rankSelector(new RankSelector<Real,true>(maxRank, this->targetRank));
                boost::shared_ptr<MinimumSelector<Real>> minSelector(new MinimumSelector<Real>());
                boost::shared_ptr<ChainSelector<Real>> selector(new ChainSelector<Real>());
                selector->add(minSelector);
                selector->add(rankSelector);
                
                IncrementalKernelEVD<DenseMatrix<Scalar>> * builder  = new IncrementalKernelEVD<DenseMatrix<Scalar>>(); 
                builder->setSelector(selector);
                builder->setPreImagesPerRank(this->targetPreImageRatio, this->maxPreImageRatio);
                builder->setUseLinearCombination(this->useLinearCombination);
                
                return builder;
            }
            
        };
        
        template<typename Scalar> struct BuilderChooser;
        
        template<typename Scalar> 
        struct DivideAndConquerConfigurator : public BuilderConfigurator<Scalar> {
            boost::scoped_ptr<BuilderConfigurator<Scalar>> builder, merger;
            typedef typename Eigen::NumTraits<Scalar>::Real Real;

            virtual std::string getName() const { return "divide-and-conquer"; }

            virtual bool processOption(std::deque<std::string> &options, std::deque<std::string> &args) override {
                if (options[0] != "merger" && options[0] != "builder")
                    return false;
                boost::scoped_ptr<BuilderConfigurator<Scalar>> &cf = options[0] == "builder" ? builder : merger;

                if (options.size() == 1) {
                    if (args.size() >= 2) {
                        args.pop_front();
                        cf.reset(BuilderChooser<Scalar>().getBuilder(args[0]));
                        args.pop_front();
                    }
                } else {
                    options.pop_front();
                    return cf->processOption(options, args);
                }
                
                return false;
            }
            
            virtual void print(std::string prefix = "") override {
                builder->print(prefix + ".builder");
                merger->print(prefix + ".merger");
            }

            boost::shared_ptr<ChainSelector<Real>>  getCleaner(const BuilderConfigurator<Scalar> &bc) {
                boost::shared_ptr<RankSelector<Real, true>> rankSelector(new RankSelector<Real,true>(bc->targetRank, bc->targetRank));
                boost::shared_ptr<MinimumSelector<Real>> minSelector(new MinimumSelector<Real>());
                boost::shared_ptr<ChainSelector<Real>> selector(new ChainSelector<Real>());
                selector->add(minSelector);
                selector->add(rankSelector);
                
                return selector;
            }
            
            virtual KernelEVD<DenseMatrix<Scalar>> *getBuilder(const KernelEVDBenchmark &) override {                

                DivideAndConquerBuilder<DenseMatrix<Scalar>> dc = new DivideAndConquerBuilder<DenseMatrix<Scalar>>();
                dc->setBuilder(builder);
                dc->setBuilderCleaner(getCleaner(*builder));
                dc->setMerger(merger);
                dc->setMergerCleaner(getCleaner(*merger));
            }
        };

        
        
        // Builder chooser (scalar dependent)
        struct BuilderChooserBase {
            virtual BuilderConfiguratorBase * getBuilder(std::string &name) = 0;
            
        };
        
        template<typename Scalar>
        struct BuilderChooser : public BuilderChooserBase {
            BuilderConfigurator<Scalar> * getBuilder(std::string &kevdName) override {
                typedef typename Eigen::NumTraits<Scalar>::Real Real;
                
                if (kevdName == "direct") 
                    return new DirectConfigurator<Scalar>();
                
                if (kevdName == "incremetal")
                    return new IncrementalConfigurator<Scalar>();
                
                if (kevdName == "accumulator") 
                    return new AccumulatorConfigurator<Scalar>();
                
                if (kevdName == "incremental") {
                    IncrementalKernelEVD<DenseMatrix<Scalar>> builder;
                    
                }
                
                KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Unknown kernel-evd builder type [%s]", %kevdName);
                
            }
        };
        
        int process(std::deque<std::string> &args) {
           
            const std::string builderPrefix = "--builder";
            
            boost::scoped_ptr<BuilderChooserBase> builderChooser;
            boost::scoped_ptr<BuilderConfiguratorBase> builderConfigurator;
            std::string scalarName = "double";
            
            // Read the arguments
            while (args.size() > 0) {
                
                if (args[0] == "--seed" && args.size() >= 2) {
                    args.pop_front();
                    seed = std::atol(args[0].c_str());
                    args.pop_front();
                } 
                
                else if (args[0] == "--scalar" && args.size() >= 2) {
                    args.pop_front();
                     scalarName = args[0];
                    args.pop_front();
                    
                    if (scalarName == "double") builderChooser.reset(new BuilderChooser<double>());
                    
                    else KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Unknown scalar type [%s]", %scalarName);
                    
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
                
                
                else if (args[0] == "--builder" && args.size() >= 2) {
                    args.pop_front();
                    if (!builderChooser) {
                        KQP_LOG_INFO(logger, "Choosing default scalar: double");
                        builderChooser.reset(new BuilderChooser<double>());   
                    }                   
                    builderConfigurator.reset(builderChooser->getBuilder(args[0]));
                    args.pop_front();
                }
                
                else if (boost::starts_with(args[0], builderPrefix)) {
                    if (!builderConfigurator) 
                        KQP_THROW_EXCEPTION(illegal_argument_exception, "The builder was not given before a --builder... option");
                    
                    std::deque<std::string> options;
                    std::string optionName = args[0].substr(builderPrefix.size());
                    boost::split(options, optionName, boost::is_any_of("-"), boost::token_compress_off);
                    if (!builderConfigurator->processOption(options, args))
                        break;
                    
                    args.pop_front();
                }
                
                else break;
            }
            
            
            if (args.size() > 0) 
                KQP_THROW_EXCEPTION_F(illegal_argument_exception, "There are %d unprocessed command line arguments, starting with [%s]", %args.size() %args[0]);
            
            // Outputs the paramaters
            std::cout << "dimension\t" << dimension << std::endl;
            std::cout << "updates\t" << updates << std::endl;
            std::cout << "seed\t" << seed << std::endl;
            std::cout << "scalar\t" << scalarName << std::endl;


            builderConfigurator->print();
            
            return builderConfigurator->run(*this);
            

        }
        
        
        
        
    };
    
    int bm_kernel_evd(std::deque<std::string> &args) {
        return KernelEVDBenchmark().process(args);
    }
}