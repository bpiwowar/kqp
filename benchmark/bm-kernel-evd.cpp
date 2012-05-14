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
#include <iostream>
#include <ctime>
#include <deque>

#include <boost/random/inversive_congruential.hpp>
#include <boost/random/uniform_01.hpp>

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
        long seed;
        
        // Space dimension
        Index dimension;
        Index updates;
        
        // Use linear combination
        bool useLC;
        
        // --- Base vector
        
        // Number of vectors use to generate new ones (uniform distribution for each component)
        Index nbVectors;
               
        // Noise ratio for generated vector components
        float noise;
        
        // --- Settings for the generation
        
        // Range for the number of pre-images at each update
        Index min_preimages;
        Index max_preimages;
        
        // Range for the number of vectors at each update
        Index min_lc;
        Index max_lc;
        
        KernelEVDBenchmark() :
        seed(0), 
        dimension(100),
        updates(1000),
        useLC(true),
        nbVectors(0),
        noise(0),
        min_preimages(1),
        max_preimages(1),
        min_lc(1),
        max_lc(1)
        {}
        
        
          
        
        
        // --- 
        
        struct BuilderConfiguratorBase {
            Index targetRank;
            float targetPreImageRatio;
            
            BuilderConfiguratorBase() :
                targetRank(std::numeric_limits<Index>::max()),
                targetPreImageRatio(std::numeric_limits<float>::infinity())
            {
                
            }
            
            virtual void print(const KernelEVDBenchmark &bm, const std::string & prefix = "") const {
                std::cout << prefix << "\t" << this->getName() << std::endl;
                std::cout << prefix << ".rank\t" << targetRank << std::endl;
                if (!bm.useLC) std::cout << prefix << ".pre_images\t" << targetPreImageRatio << std::endl;
            }
            
            virtual std::string getName() const = 0;
            virtual bool processOption(std::deque<std::string> &options, std::deque<std::string> &args) {
                if (options.size() == 1 && options[0] == "rank" && args.size() >= 2) {
                    args.pop_front();
                    targetRank = boost::lexical_cast<Index>(args[0]);
                    args.pop_front();
                    return true;
                }
                
                if (options.size() == 1 && options[0] == "pre_images" && args.size() >= 2) {
                    args.pop_front();
                    targetPreImageRatio = boost::lexical_cast<float>(args[0]);
                    args.pop_front();
                    return true;
                }

                return false;
            }
            
            virtual int run(const KernelEVDBenchmark &) = 0;
        };
        
        template<typename _Scalar>
        struct BuilderConfigurator : public BuilderConfiguratorBase {
            typedef _Scalar Scalar;
            KQP_SCALAR_TYPEDEFS(Scalar);
            
            typedef Dense<Scalar> KQPMatrix;
            typedef DenseSpace<Scalar> KQPSpace;
            typedef boost::shared_ptr<Cleaner<Real>> CleanerPtr;

            virtual KernelEVD<Scalar>  *getBuilder(const Space<Scalar> &, const KernelEVDBenchmark &) = 0;

            ScalarMatrix m_genVectors;
            
            boost::hellekalek1995 generator;
            boost::uniform_01<double> uniformGenerator; 
            
            //! Initialise
            void init(const KernelEVDBenchmark &bm) {
                generator.seed(bm.seed);
                
                if (bm.nbVectors > 0) {
                    KQP_LOG_INFO_F(logger, "Creating %d base vectors in dimension %d", %bm.nbVectors %bm.dimension);
                    m_genVectors = ScalarMatrix::Random(bm.dimension, bm.nbVectors);
                }
            }
            
            //! Get the next feature matrix + mixture matrix
            void getNext(const KernelEVDBenchmark &bm, Real &alpha, ScalarMatrix &m, ScalarMatrix &mA) const {
                alpha = Eigen::internal::abs(Eigen::internal::random_impl<Real>::run()) + 1e-3;
            
                int k = (int)std::abs(Eigen::internal::random_impl<double>::run() * (double)(bm.max_preimages-bm.min_preimages)) + bm.min_preimages;
                int p = (int)std::abs(Eigen::internal::random_impl<double>::run() * (double)(bm.max_lc-bm.min_lc)) + bm.min_lc;
                KQP_LOG_DEBUG(logger, boost::format("Pre-images (%dx%d) and linear combination (%dx%d)") % bm.dimension % k % k % p);
            
                // Generate the linear combination matrix
                mA = ScalarMatrix::Random(k, p);
            
                // Generate k pre-images
                if (bm.nbVectors > 0) {
                    // TODO: weights should be generated according to some distribution
                    RealVector weights = RealVector::Random(bm.nbVectors);
                    
                    m = m_genVectors * weights.asDiagonal() * ScalarMatrix::Random(bm.nbVectors,k) + bm.noise * ScalarMatrix::Random(bm.dimension, k);
                } else {
                    m = ScalarMatrix::Random(bm.dimension, k);         
                }
            }

            
            int run(const KernelEVDBenchmark &bm) {
                
                FSpace fs = KQPSpace::create(bm.dimension);
                fs.setUseLinearCombination(bm.useLC);
                init(bm);
                
                boost::scoped_ptr<KernelEVD<Scalar>> builder(this->getBuilder(fs, bm));
                
                std::clock_t total_time = 0;
                
                // --- Computing Kernel EVD
                
                KQP_LOG_INFO_F(logger, "Computing kernel EVD with builder %s", % KQP_DEMANGLE(*builder));
                
                std::clock_t c_start = std::clock();
                
                std::srand(bm.seed);
                ScalarMatrix mU, mA;
                Real alpha;
                for(int i = 0; i < bm.updates; i++) {
                    getNext(bm, alpha, mU, mA);
                    builder->add(alpha, KQPMatrix::create(mU), mA);
                }        
                
                Decomposition<Scalar> result = builder->getDecomposition();
                std::clock_t c_end = std::clock();
                total_time += c_end-c_start;
                std::cout << "kevd\t" << 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC << std::endl;
                KQP_LOG_INFO_F(logger, "Decomposition result: %d pre-images and rank %d", %result.mX.size() %result.mY.cols());
                
                // --- Cleaning up
                
                KQP_LOG_INFO(logger, "Cleaning up");
                c_start = std::clock();
                CleanerList<Real> cleaner;
                
                boost::shared_ptr<RankSelector<Real, true>> rankSelector(new RankSelector<Real,true>(this->targetRank));
                boost::shared_ptr<MinimumSelector<Real>> minSelector(new MinimumSelector<Real>(Eigen::NumTraits<Real>::epsilon()));
                boost::shared_ptr<ChainSelector<Real>> selector(new ChainSelector<Real>());
                selector->add(minSelector);
                selector->add(rankSelector);
                
                cleaner.add(CleanerPtr(new CleanerRank<Scalar>(rankSelector)));
                cleaner.add(CleanerPtr(new CleanerUnused<Scalar>()));
                
                boost::shared_ptr<CleanerQP<Real>> qpCleaner(new CleanerQP<Scalar>());
                qpCleaner->setPreImagesPerRank(this->targetPreImageRatio, this->targetPreImageRatio);
                cleaner.add(qpCleaner);
                
                cleaner.cleanup(result);
                
                c_end = std::clock();
                total_time += c_end-c_start;
                std::cout << "cleaning\t" << 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC << std::endl;
                
                if (!result.orthonormal) {
                    KQP_LOG_INFO(logger, "Re-orthonormalizing");
                    c_start = std::clock();
                    Orthonormalize<Scalar>::run(fs, result.mX, result.mY, result.mD);
                    
                    c_end = std::clock();
                    total_time += c_end-c_start;
                    std::cout << "orthonormalizing\t" << 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC << std::endl;
                } else {
                    std::cout << "orthonormalizing\t" << 0 << std::endl;
                }
                
                std::cout << "time\t" << 1000.0 * (total_time) / CLOCKS_PER_SEC << std::endl;
                
                // --- Computing the error
                KQP_LOG_INFO(logger, "Computing the error");
                
                double orthogononal_error = 0;            
                ScalarMatrix sumWWT(result.mY.cols(), result.mY.cols());
                sumWWT.setZero();
                
                ScalarMatrix mYTXT = result.mY.adjoint() * result.mX->template as<Dense<Scalar>>()->adjoint();
                
                std::srand(bm.seed);
                for(int i = 0; i < bm.updates; i++) {
                    // Get the next matrix
                    getNext(bm, alpha, mU, mA);
                    
                    // Project
                    ScalarMatrix mW = mYTXT * mU * mA;
                    sumWWT.template selfadjointView<Eigen::Lower>().rankUpdate(mW, alpha);
                    orthogononal_error += Eigen::internal::abs(alpha) * (mA.adjoint() * mU.adjoint() * mU * mA - mW.adjoint() * mW).squaredNorm();
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
            virtual KernelEVD<Scalar> *getBuilder(const Space<Scalar> &, const KernelEVDBenchmark &bm) override {
                return new DenseDirectBuilder<Scalar>(bm.dimension);
            };
            virtual std::string getName() const { return "direct"; }
        };
        
        template<typename Scalar> 
        struct AccumulatorConfigurator : public BuilderConfigurator<Scalar> {
            
            virtual KernelEVD<Scalar> *getBuilder(const Space<Scalar> &fs, const KernelEVDBenchmark &) override {
                if (fs.canLinearlyCombine()) 
                    return new AccumulatorKernelEVD<Scalar,true>(fs);
                
                KQP_LOG_INFO(logger, "Accumulator without linear combination selected");
                return new AccumulatorKernelEVD<Scalar,false>(fs);
            }
            
            virtual std::string getName() const {
                return "accumulator";
            }
        };
  
        
        template<typename Scalar> 
        struct IncrementalConfigurator : public BuilderConfigurator<Scalar> {
            typedef typename Eigen::NumTraits<Scalar>::Real Real;
            float maxPreImageRatio;
            Index maxRank;
            
            IncrementalConfigurator() : maxPreImageRatio(std::numeric_limits<float>::infinity()), maxRank(-1) {}
            
            virtual void print(const KernelEVDBenchmark &bm, const std::string& prefix = "") const override {
                BuilderConfigurator<Scalar>::print(bm, prefix);
                if (!bm.useLC) std::cout << prefix << ".max_pre_images\t" << maxPreImageRatio << std::endl;
                std::cout << prefix << ".max_rank\t" << maxRank << std::endl;
            }
            
            virtual bool processOption(std::deque<std::string> &options, std::deque<std::string> &args) override {
                if (options.size() == 1) {
                    if (options[0] == "max_rank" && args.size() >= 2) {
                        args.pop_front();
                        maxRank = boost::lexical_cast<Index>(args[0]);
                        args.pop_front();
                        return true;
                    }
                    
                    if (options[0] == "max_pre_images" && args.size() >= 2) {
                        args.pop_front();
                        maxPreImageRatio = boost::lexical_cast<float>(args[0]);
                        args.pop_front();
                        return true;
                    }
                }
                
                return BuilderConfigurator<Scalar>::processOption(options, args);            
            }
        
            virtual std::string getName() const { return "incremental"; }
            
            virtual KernelEVD<Scalar> *getBuilder(const Space<Scalar> &fs, const KernelEVDBenchmark &) override {
                // Construct the rank selector
                boost::shared_ptr<RankSelector<Real, true>> rankSelector(new RankSelector<Real,true>(maxRank, this->targetRank));
                boost::shared_ptr<MinimumSelector<Real>> minSelector(new MinimumSelector<Real>(Eigen::NumTraits<Real>::epsilon()));
                boost::shared_ptr<ChainSelector<Real>> selector(new ChainSelector<Real>());
                selector->add(minSelector);
                selector->add(rankSelector);
                
                IncrementalKernelEVD<Scalar> * builder  = new IncrementalKernelEVD<Scalar>(fs); 
                builder->setSelector(selector);
                builder->setPreImagesPerRank(this->targetPreImageRatio, this->maxPreImageRatio);
                
                return builder;
            }
            
        };
        
        template<typename Scalar> struct BuilderChooser;
        
        
        /** Configurator for "Divide and Conquer" KEVD */
        template<typename Scalar> 
        struct DivideAndConquerConfigurator : public BuilderConfigurator<Scalar> {
            typedef typename Eigen::NumTraits<Scalar>::Real Real;
            typedef boost::shared_ptr<KernelEVD<Scalar>> KEVDPtr;
            typedef boost::shared_ptr<Cleaner<Real>> CleanerPtr;

            boost::scoped_ptr<BuilderConfigurator<Scalar>> builder, merger;
            Index batchSize;
            
            virtual std::string getName() const { return "divide-and-conquer"; }

            DivideAndConquerConfigurator() : batchSize(100) {}
            
            virtual bool processOption(std::deque<std::string> &options, std::deque<std::string> &args) override {
                
                if (options.size() == 2 && options[0] == "batch" && options[1] == "size" && args.size() >= 2) {
                    batchSize = boost::lexical_cast<Index>(args[1]);
                    args.pop_front();
                    args.pop_front();
                    return true;
                }
                
                if (options[0] != "merger" && options[0] != "builder")
                    return false;
                boost::scoped_ptr<BuilderConfigurator<Scalar>> &cf = options[0] == "builder" ? builder : merger;

                
                if (options.size() == 1) {
                    if (args.size() >= 2) {
                        args.pop_front();
                        cf.reset(BuilderChooser<Scalar>().getBuilder(args[0]));
                        KQP_LOG_INFO_F(logger, "%s for D&C is %s", %options[0] %args[0]);
                        args.pop_front();
                        return true;
                    }
                } else {
                    options.pop_front();
                    return cf->processOption(options, args);
                }
                
                return false;
            }
            
            virtual void print(const KernelEVDBenchmark &bm, const std::string & prefix = "") const override {
                BuilderConfigurator<Scalar>::print(bm, prefix);
                std::cout << prefix << ".batch_size\t" << batchSize << std::endl;
                builder->print(bm, prefix + ".builder");
                merger->print(bm, prefix + ".merger");
            }

            boost::shared_ptr<Cleaner<Real>>  getCleaner(const KernelEVDBenchmark &, const BuilderConfigurator<Scalar> &bc) {
                // Construct the rank selector
                boost::shared_ptr<RankSelector<Real, true>> rankSelector(new RankSelector<Real,true>(bc.targetRank, bc.targetRank));
                boost::shared_ptr<MinimumSelector<Real>> minSelector(new MinimumSelector<Real>(Eigen::NumTraits<Scalar>::epsilon()));
                boost::shared_ptr<ChainSelector<Real>> selector(new ChainSelector<Real>());
                selector->add(minSelector);
                selector->add(rankSelector);
                
                // Construct the cleaner
                boost::shared_ptr<CleanerList<Scalar>> cleaner(new CleanerList<Scalar>());
                
                cleaner->add(CleanerPtr(new CleanerRank<Scalar>(rankSelector)));
                cleaner->add(CleanerPtr(new CleanerUnused<Scalar>()));
                
                boost::shared_ptr<CleanerQP<Real>> qpCleaner(new CleanerQP<Scalar>());
                qpCleaner->setPreImagesPerRank(bc.targetPreImageRatio, bc.targetPreImageRatio);
                cleaner->add(qpCleaner);
                
                return cleaner;

            }
            
            virtual KernelEVD<Scalar> *getBuilder(const Space<Scalar> &fs, const KernelEVDBenchmark &bm) override {                
                DivideAndConquerBuilder<Scalar> *dc = new DivideAndConquerBuilder<Scalar>(fs);
                dc->setBatchSize(batchSize);
                
                dc->setBuilder(KEVDPtr(builder->getBuilder(fs, bm)));
                dc->setBuilderCleaner(getCleaner(bm, *builder));
                
                dc->setMerger(KEVDPtr(merger->getBuilder(fs,bm)));
                dc->setMergerCleaner(getCleaner(bm, *merger));
                return dc;
            }
        };

        
        
        // Builder chooser (scalar dependent)
        struct BuilderChooserBase {
            virtual BuilderConfiguratorBase * getBuilder(std::string &name) = 0;
            
        };
        
        /** Chooser for the Kernel EVD algorithm */
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
                
                if (kevdName == "incremental") 
                    return new IncrementalConfigurator<Scalar>();

                if (kevdName == "divide-and-conquer") 
                    return new DivideAndConquerConfigurator<Scalar>();

                KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Unknown kernel-evd builder type [%s]", %kevdName);
                
            }
        };
        
        int process(std::deque<std::string> &args) {
           
            const std::string builderPrefix = "--builder-";
            
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

                else if (args[0] == "--no-lc") {
                    args.pop_front();
                    useLC = false;
                } 

                else if (args[0] == "--noise" && args.size() >= 2) {
                    args.pop_front();
                    noise = boost::lexical_cast<double>(args[0]);
                    args.pop_front();
                } 

                else if (args[0] == "--nb-vectors" && args.size() >= 2) {
                    args.pop_front();
                    nbVectors = boost::lexical_cast<Index>(args[0]);
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
                    
                }
                
                else break;
            }
            
            
          
            if (args.size() > 0) 
                KQP_THROW_EXCEPTION_F(illegal_argument_exception, "There are %d unprocessed command line arguments, starting with [%s]", %args.size() %args[0]);
            
            // Outputs the paramaters
            std::cout << "dimension\t" << dimension << std::endl;
            std::cout << "updates\t" << updates << std::endl;
            std::cout << "lc\t" << (useLC ? "true" : "false")  << std::endl;
            std::cout << "seed\t" << seed << std::endl;
            std::cout << "scalar\t" << scalarName << std::endl;
            std::cout << "nb_v\t" << nbVectors << std::endl;
            std::cout << "noise\t" << noise << std::endl;

            if (!builderConfigurator.get()) 
                KQP_THROW_EXCEPTION(illegal_argument_exception, "No builder was given (with --builder)");
                
            builderConfigurator->print(*this, "builder");
            
            return builderConfigurator->run(*this);
            

        }
        
        
        
        
    };
    
    int bm_kernel_evd(std::deque<std::string> &args) {
        return KernelEVDBenchmark().process(args);
    }
}