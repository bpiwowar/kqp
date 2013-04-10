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

#include <fstream>
#include <ctime>
#include <deque>
#include <iterator>

#include <boost/random/inversive_congruential.hpp>
#include <boost/random/uniform_01.hpp>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/algorithm/string.hpp>

#include <kqp/picojson.h>

#include <kqp/kqp.hpp>
#include <kqp/cleanup.hpp>

#include <kqp/kernel_evd/dense_direct.hpp>
#include <kqp/kernel_evd/accumulator.hpp>
#include <kqp/kernel_evd/incremental.hpp>
#include <kqp/kernel_evd/divide_and_conquer.hpp>

DEFINE_LOGGER(logger,  "kqp.benchmark.kernel-evd");



namespace kqp {
	namespace {

		template<typename type>
		type get(const std::string &context, picojson::object &o, const std::string &key, const type & _default) {
			if (o.find(key) == o.end()) {
				o[key] = picojson::value(_default);
				return _default;
			}
			
			if (!o[key].is<type>())
				KQP_THROW_EXCEPTION_F(kqp::illegal_argument_exception, "JSON error [%s]: [%s] is not of type %s", %context %key %KQP_STRING_IT(TYPE));
			return o[key].get<type>();
		}

		
		template<typename type>
		type get(const std::string &context, picojson::value &d, const std::string &key, const type & _default) {
			if (!d.is<picojson::object>())
				KQP_THROW_EXCEPTION_F(kqp::illegal_argument_exception, "JSON error [%s]: not an object", %context);
			
			picojson::object o = d.get<picojson::object>();
			return get(context, o, key, _default);
		}
		
		template<typename type>
		type get(const std::string &context, picojson::object &o, const std::string &key) {
			if (o.find(key) == o.end()) {
				KQP_THROW_EXCEPTION_F(kqp::illegal_argument_exception, "JSON error [%s]: no key [%s]", %context %key );
			}
			
			if (!o[key].is<type>())
				KQP_THROW_EXCEPTION_F(kqp::illegal_argument_exception, "JSON error [%s]: [%s] is not of type %s", %context %key %KQP_STRING_IT(TYPE));
			return o[key].get<type>();
		}

		template<typename type>
		type get(const std::string &context, picojson::value &d, const std::string &key) {
			if (!d.is<picojson::object>())
				KQP_THROW_EXCEPTION_F(kqp::illegal_argument_exception, "JSON error [%s]: not an object", %context);
			
			picojson::object o = d.get<picojson::object>();
			return get<type>(context, o, key);
		}
		
		template<>
		int get(const std::string &context, picojson::value &d, const std::string &key, const int & _default) {
			return boost::lexical_cast<int>(get<double>(context, d, key, (double)_default));
		}
		template<>
		int get(const std::string &context, picojson::object &o, const std::string &key, const int & _default) {
			return boost::lexical_cast<int>(get<double>(context, o, key, (double)_default));
		}
		
		template<>
		int get(const std::string &context, picojson::value &d, const std::string &key) {
			return boost::lexical_cast<int>(get<double>(context, d, key));
		}
		template<>
		int get(const std::string &context, picojson::object &o, const std::string &key) {
			return boost::lexical_cast<int>(get<double>(context, o, key));
		}
		
	}
	
    // FIXME: use boost program options library instead
    
    
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
            
            BuilderConfiguratorBase()
            {
                
            }
            
            virtual std::string getName() const = 0;
			
            virtual void configure(const KernelEVDBenchmark &bm, const std::string &context, picojson::object &json) {
				(void)bm; (void)context; (void)json;
            }
			
			virtual void setCleaner(const std::string &context, picojson::value &json) = 0;
            
            virtual int run(picojson::object &value, const KernelEVDBenchmark &bm) = 0;
        };
        
        template<typename _Scalar>
        struct BuilderConfigurator : public BuilderConfiguratorBase {
            typedef _Scalar Scalar;
            KQP_SCALAR_TYPEDEFS(Scalar);
			
            typedef boost::shared_ptr<Selector<Real>> SelectorPtr;
			typedef boost::shared_ptr<Cleaner<Scalar>> CleanerPtr;
			
            typedef Dense<Scalar> KQPMatrix;
            typedef DenseSpace<Scalar> KQPSpace;
			
			CleanerPtr m_cleaner;
			
            virtual KernelEVD<Scalar>  *getBuilder(const FSpaceCPtr &, const KernelEVDBenchmark &) = 0;
			
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
			
			/** Sets the cleaner */
			virtual void setCleaner(const std::string &context, picojson::value &json) override {
				m_cleaner = getCleaner(context, json);
			}
			
			virtual SelectorPtr getSelector(const std::string &context, picojson::value &json) {
				boost::shared_ptr<ChainSelector<Real>> chain(new ChainSelector<Real>());
				if (json.is<picojson::null>())
					return chain;
				if (!json.is<picojson::array>())
					KQP_THROW_EXCEPTION_F(kqp::illegal_argument_exception, "Error in JSON [%s]: should be an array", %context);
				
				picojson::array array = json.get<picojson::array>();
				for(size_t i = 0; i < array.size(); i++) {
					auto &jsonSel = array[i];
					std::string contextSel = context + "[" + boost::lexical_cast<std::string>(i) + "]";
					std::string name = get<std::string>(contextSel, jsonSel, "name");
					if (name == "rank") {
						int maxRank = get<int>(contextSel, jsonSel, "max");
						int resetRank = get<int>(contextSel, jsonSel, "reset");
						chain->add(SelectorPtr(new RankSelector<Real,true>(maxRank, resetRank)));
					} else if (name == "ratio") {
						typename RatioSelector<Real>::AggregatorPtr aggregator(new Mean<Real>());
						Real ratio = (Real)get<double>(contextSel, jsonSel, "ratio", Eigen::NumTraits<Real>::epsilon());
						chain->add(SelectorPtr(new RatioSelector<Real>(ratio, aggregator)));
					} else KQP_THROW_EXCEPTION_F(kqp::illegal_argument_exception, "Unknown selector [%s] in [%s]", %name %context);
				}
				
				return chain;
			}
			
			virtual boost::shared_ptr<Cleaner<Scalar>> getCleaner(const std::string &context, picojson::value &json) override {
				boost::shared_ptr<CleanerList<Real>> list(new CleanerList<Real>());
				if (json.is<picojson::null>())
					return list;
				
				if (!json.is<picojson::array>())
					KQP_THROW_EXCEPTION_F(kqp::illegal_argument_exception, "Error in JSON [%s]: should be an array", %context);
				
				picojson::array array = json.get<picojson::array>();
				
				for(size_t i = 0; i < array.size(); i++) {
					auto &jsonCleaner = array[i];
					std::string contextCleaner = context + "[" + boost::lexical_cast<std::string>(i) + "]";
					std::string name = get<std::string>(contextCleaner, jsonCleaner, "name");
					if (name == "rank") {
						auto rankSelector = getSelector(contextCleaner + ".selector", jsonCleaner.get<picojson::object>()["selector"]);
						list->add(CleanerPtr(new CleanerRank<Scalar>(rankSelector)));
					} else if (name == "unused") {
						list->add(CleanerPtr(new CleanerUnused<Scalar>()));
					} else if (name == "null") {
						boost::shared_ptr<CleanerNullSpace<Scalar>> cleaner(new CleanerNullSpace<Scalar>());
						cleaner->epsilon(get<double>(contextCleaner, jsonCleaner, "epsilon", Eigen::NumTraits<Real>::epsilon()));
						
						double maxRatio = get<double>(contextCleaner, jsonCleaner, "max-ratio",  std::numeric_limits<Real>::infinity());
						double resetRatio = get<double>(contextCleaner, jsonCleaner, "reset-ratio", 0);
						cleaner->setPreImagesPerRank(resetRatio, maxRatio);

						int maxRank = get<int>(contextCleaner, jsonCleaner, "max-rank", std::numeric_limits<Index>::max());
						int resetRank = get<int>(contextCleaner, jsonCleaner, "reset-rank", 0);
						cleaner->setRankRange(resetRank, maxRank);
						
						list->add(cleaner);
					} else if (name == "qp") {
						boost::shared_ptr<CleanerQP<Real>> cleaner(new CleanerQP<Scalar>());

						double maxRatio = get<double>(contextCleaner, jsonCleaner, "max-ratio",  std::numeric_limits<Real>::infinity());
						double resetRatio = get<double>(contextCleaner, jsonCleaner, "reset-ratio", 0);
						cleaner->setPreImagesPerRank(resetRatio, maxRatio);
						
						int maxRank = get<int>(contextCleaner, jsonCleaner, "max-rank", std::numeric_limits<Index>::max());
						int resetRank = get<int>(contextCleaner, jsonCleaner, "reset-rank", 0);
						cleaner->setRankRange(resetRank, maxRank);
						
						list->add(cleaner);
					} else KQP_THROW_EXCEPTION_F(kqp::illegal_argument_exception, "Unknown cleaner [%s] in [%s]", %name %context);
					
				}
                
				return list;
			}
			
            int run(picojson::object &value, const KernelEVDBenchmark &bm) override {
                
                FSpace fs = KQPSpace::create(bm.dimension);
                fs->setUseLinearCombination(bm.useLC);
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
                

				picojson::object &times = (value["time"] = picojson::value(picojson::object())).get<picojson::object>();
				picojson::object &errors = (value["error"] = picojson::value(picojson::object())).get<picojson::object>();
				
                Decomposition<Scalar> result = builder->getDecomposition();
                std::clock_t c_end = std::clock();
                total_time += c_end-c_start;
				times["kevd"] = picojson::value(1000.0 * (c_end-c_start) / CLOCKS_PER_SEC);
                KQP_LOG_INFO_F(logger, "Decomposition result: %d pre-images and rank %d", %result.mX->size() %result.mY.cols());
                
                // --- Cleaning up
                
                KQP_LOG_INFO(logger, "Cleaning up");
                c_start = std::clock();
                
                m_cleaner->cleanup(result);
                
                c_end = std::clock();
                total_time += c_end-c_start;
				times["cleaning"] = picojson::value(1000.0 * (c_end-c_start) / CLOCKS_PER_SEC);
                
                if (!result.orthonormal) {
                    KQP_LOG_INFO(logger, "Re-orthonormalizing");
                    c_start = std::clock();
                    Orthonormalize<Scalar>::run(fs, result.mX, result.mY, result.mD);
                    
                    c_end = std::clock();
                    total_time += c_end-c_start;
                    times["orthonormalizing"] = picojson::value(1000.0 * (c_end-c_start) / CLOCKS_PER_SEC);
                } else {
                    times["orthonormalizing"] = picojson::value(0.);
                }
                
                times["total"] = picojson::value(1000.0 * (total_time) / CLOCKS_PER_SEC);
                
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
				
                errors["o_error"] = picojson::value(orthogononal_error);
                errors["s_error"] = picojson::value((ScalarMatrix(sumWWT.template selfadjointView<Eigen::Lower>()) - ScalarMatrix(result.mD.asDiagonal())).squaredNorm());
                errors["pre_images"] = picojson::value((double)result.mX->size());
                errors["rank"] = picojson::value((double)result.mY.cols());
				

                return 0;
                
            }
            
        };
        
        
        template<typename Scalar>
        struct DirectConfigurator : public BuilderConfigurator<Scalar> {
            KQP_SCALAR_TYPEDEFS(Scalar);
            virtual KernelEVD<Scalar> *getBuilder(const FSpaceCPtr &, const KernelEVDBenchmark &bm) override {
                return new DenseDirectBuilder<Scalar>(bm.dimension);
            };
            virtual std::string getName() const { return "direct"; }
        };
        
        template<typename Scalar>
        struct AccumulatorConfigurator : public BuilderConfigurator<Scalar> {
            KQP_SCALAR_TYPEDEFS(Scalar);
            
            virtual KernelEVD<Scalar> *getBuilder(const FSpaceCPtr &fs, const KernelEVDBenchmark &) override {
                if (fs->canLinearlyCombine())
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
            KQP_SCALAR_TYPEDEFS(Scalar);
			boost::shared_ptr<Selector<Real>> m_selector;
            float targetPreImageRatio, maxPreImageRatio;
			
            IncrementalConfigurator() :
			targetPreImageRatio(std::numeric_limits<float>::infinity()),
			maxPreImageRatio(std::numeric_limits<float>::infinity())
			{}
            
            virtual std::string getName() const { return "incremental"; }
			
			
			virtual void configure(const KernelEVDBenchmark &bm, const std::string &context, picojson::object &json) override {
				BuilderConfigurator<Scalar>::configure(bm, context, json);
				m_selector = this->getSelector(context + ".selector", json["selector"]);
				
				if (!bm.useLC) {
					targetPreImageRatio = get<int>(context, json, "pre-images");
					maxPreImageRatio = get<int>(context, json, "max-pre-images", targetPreImageRatio);
				}
			}
			
            virtual KernelEVD<Scalar> *getBuilder(const FSpaceCPtr &fs, const KernelEVDBenchmark &) override {
                IncrementalKernelEVD<Scalar> * builder  = new IncrementalKernelEVD<Scalar>(fs);
                builder->setSelector(m_selector);
                builder->setPreImagesPerRank(this->targetPreImageRatio, this->maxPreImageRatio);
                
                return builder;
            }
            
        };
        
        template<typename Scalar> struct BuilderChooser;
        
        
        /** Configurator for "Divide and Conquer" KEVD */
        template<typename Scalar>
        struct DivideAndConquerConfigurator : public BuilderConfigurator<Scalar> {
            KQP_SCALAR_TYPEDEFS(Scalar);
            typedef boost::shared_ptr<KernelEVD<Scalar>> KEVDPtr;
            typedef boost::shared_ptr<Cleaner<Real>> CleanerPtr;
			
			typedef boost::shared_ptr<BuilderConfigurator<Scalar>> BuilderPtr;
            BuilderPtr builder, merger;
			CleanerPtr builderCleaner, mergerCleaner;
			
            Index batchSize;
            
            virtual std::string getName() const { return "divide-and-conquer"; }
			
            DivideAndConquerConfigurator() : batchSize(100) {}
            
            virtual void configure(const KernelEVDBenchmark &bm, const std::string &context, picojson::object &json) override {
				BuilderConfigurator<Scalar>::configure(bm, context, json);
				
                batchSize = get<int>(context, json, "batch-size", batchSize);
				
				builder.reset(BuilderChooser<Scalar>().getBuilder(bm, context + ".builder", json["builder"]));
				merger.reset(BuilderChooser<Scalar>().getBuilder(bm, context + ".merger", json["merger"]));
				
				builderCleaner = this->getCleaner(context + ".builder.cleaner", json["builder"].get<picojson::object>()["cleaner"]);
				mergerCleaner = this->getCleaner(context + ".merger.cleaner", json["merger"].get<picojson::object>()["cleaner"]);
			}
            
            virtual KernelEVD<Scalar> *getBuilder(const FSpaceCPtr &fs, const KernelEVDBenchmark &bm) override {
                DivideAndConquerBuilder<Scalar> *dc = new DivideAndConquerBuilder<Scalar>(fs);
                dc->setBatchSize(batchSize);
				
                
                dc->setBuilder(KEVDPtr(builder->getBuilder(fs, bm)));
                dc->setBuilderCleaner(builderCleaner);
                
                dc->setMerger(KEVDPtr(merger->getBuilder(fs,bm)));
                dc->setMergerCleaner(mergerCleaner);
                return dc;
            }
        };
		
        
        
        // Builder chooser (scalar dependent)
        struct BuilderChooserBase {
            virtual BuilderConfiguratorBase * getBuilder(const KernelEVDBenchmark &bm, const std::string &context, picojson::value &json) = 0;
        };
        
        /** Chooser for the Kernel EVD algorithm */
        template<typename Scalar>
        struct BuilderChooser : public BuilderChooserBase {
            BuilderConfigurator<Scalar> * getBuilder(const KernelEVDBenchmark &bm, const std::string &context, picojson::value &json) override {
				if (json.is<picojson::null>())
					KQP_THROW_EXCEPTION_F(kqp::illegal_argument_exception, "No builder defined [%s]", %context);
				if (!json.is<picojson::object>())
					KQP_THROW_EXCEPTION_F(kqp::illegal_argument_exception, "Builder is not a JSON object", %context);
				
				std::string kevdName = get<std::string>(context, json, "name");
				
                typedef typename Eigen::NumTraits<Scalar>::Real Real;
                
				BuilderConfigurator<Scalar> *builderFactory;
				
                if (kevdName == "incremental")
                    builderFactory = new IncrementalConfigurator<Scalar>();

                else if (kevdName == "direct")
                    builderFactory = new DirectConfigurator<Scalar>();
                
                
                else if (kevdName == "accumulator")
                    builderFactory = new AccumulatorConfigurator<Scalar>();
                
                else if (kevdName == "incremental")
                    builderFactory = new IncrementalConfigurator<Scalar>();
				
                else if (kevdName == "divide-and-conquer")
                    builderFactory = new DivideAndConquerConfigurator<Scalar>();
				
                else KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Unknown kernel-evd builder type [%s]", %kevdName);
				
				builderFactory->configure(bm, context, json.get<picojson::object>());
				return builderFactory;
            }
        };
        
        int process(picojson::value &d) {
			
            boost::scoped_ptr<BuilderChooserBase> builderChooser;
            boost::scoped_ptr<BuilderConfiguratorBase> builderConfigurator;
			
			std::string name = get<std::string>("", d, "name", "na");
			
			seed = get<double>("", d, "seed", seed);
			useLC = get<bool>("", d, "lc", useLC);
			noise = get<double>("", d, "noise", noise);
			nbVectors = get<int>("", d, "nb-vectors", nbVectors);
			
			std::string scalarName = get<std::string>("", d, "scalar", "double");
			if (scalarName == "double") builderChooser.reset(new BuilderChooser<double>());
			else KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Unknown scalar type [%s]", %scalarName);
			
			dimension = get<int>("", d, "dimension", dimension);
			updates = get<int>("", d, "updates", updates);
			
			auto &o = d.get<picojson::object>();
			builderConfigurator.reset(builderChooser->getBuilder(*this, "builder", o["builder"]));
			
			if (o.find("cleaner") != o.end())
				builderConfigurator->setCleaner("cleaner", o["cleaner"]);
			
            // Outputs the paramaters
            return builderConfigurator->run(d.get<picojson::object>(), *this);
            
			
        }
        
        
        
        
    };
    
    // One argument
    int bm_kernel_evd(std::deque<std::string> &args) {
        if (args.size() != 1)
            KQP_THROW_EXCEPTION(illegal_argument_exception, "Expected one argument: a JSON file");
		
		std::ifstream file(args[0].c_str());
		picojson::value v;
		file >> v;
		
		std::string err = picojson::get_last_error();
		if (! err.empty()) {
			KQP_THROW_EXCEPTION_F(kqp::illegal_argument_exception, "JSON parsing error: %s (offset %d)", %err);
		}
		
        int code =  KernelEVDBenchmark().process(v);
		
		v.serialize(std::ostream_iterator<char>(std::cout));
		std::cout << std::endl;
		
		return code;
    }
}


