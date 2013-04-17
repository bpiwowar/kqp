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

#ifndef __KQP_KERNEL_EVD_FACTORY_H__
#define __KQP_KERNEL_EVD_FACTORY_H__

#include <kqp/cleanup.hpp>

#include <kqp/kernel_evd/dense_direct.hpp>
#include <kqp/kernel_evd/accumulator.hpp>
#include <kqp/kernel_evd/incremental.hpp>
#include <kqp/kernel_evd/divide_and_conquer.hpp>

#include <kqp/picojson.hpp>

namespace kqp {
	template<typename _Scalar> struct BuilderFactory;
	
	/**
	 * Non specialized builder factory
	 */
	struct BuilderFactoryBase {
		typedef boost::shared_ptr<BuilderFactoryBase> Ptr;

		virtual ~BuilderFactoryBase() {}
		
		//! Get a new builder
		virtual boost::shared_ptr< KernelEVDBase > getBuilder() = 0;

#ifndef SWIG
		//! Get a factory
		inline static Ptr getFactory(const boost::shared_ptr<AbstractSpace> &space, picojson::value &json);

		/**
		* Get a cleaner from JSON
		* @param context A context string for error messages
		*
		*/
		inline static boost::shared_ptr<CleanerBase> getCleaner(picojson::value &json, const std::string &context = "");

		virtual void configure(const std::string &context, picojson::object &json) {
			(void)context; (void)json;
		}
#endif
	};

#ifndef SWIG	
	template<typename Scalar> class IncrementalFactory;
	template<typename Scalar> class DirectFactory;
	template<typename Scalar> class DivideAndConquerFactory;
	template<typename Scalar> class AccumulatorFactory;
	
	template<typename _Scalar>
	struct BuilderFactory : public BuilderFactoryBase {
		typedef _Scalar Scalar;
		KQP_SCALAR_TYPEDEFS(Scalar);

		FSpaceCPtr m_space;
		
		BuilderFactory() {}

		void setSpace(const FSpaceCPtr &space) {
			this->m_space = space;
		}
		
		virtual ~BuilderFactory() {}
		
		typedef boost::shared_ptr<Selector<Real>> SelectorPtr;
		typedef boost::shared_ptr<Cleaner<Real>> CleanerPtr;
		
		virtual boost::shared_ptr<KernelEVDBase> getBuilder() = 0;
		
		static boost::shared_ptr<BuilderFactory<Scalar>> getFactory(const boost::shared_ptr<const AbstractSpace> &space, picojson::value &json, const std::string &context = "") {
			if (json.is<picojson::null>())
				KQP_THROW_EXCEPTION_F(kqp::illegal_argument_exception, "No builder defined [%s]", %context);
			if (!json.is<picojson::object>())
				KQP_THROW_EXCEPTION_F(kqp::illegal_argument_exception, "Builder is not a JSON object", %context);
			
			return BuilderFactory<Scalar>::getFactory(space, json.get<picojson::object>(), context);
		}
		
		static boost::shared_ptr<BuilderFactory<Scalar>> getFactory(const boost::shared_ptr<const AbstractSpace> &space, picojson::object &json, const std::string &context = "") {
			
			std::string kevdName = get<std::string>(context, json, "name");
			
			typedef typename Eigen::NumTraits<Scalar>::Real Real;
			
			boost::shared_ptr<BuilderFactory<Scalar>> builderFactory;
			
			if (kevdName == "incremental")
				builderFactory.reset(new IncrementalFactory<Scalar>());
			
			else if (kevdName == "direct")
				builderFactory.reset(new DirectFactory<Scalar>());
			
			else if (kevdName == "accumulator")
				builderFactory.reset(new AccumulatorFactory<Scalar>());
			
			else if (kevdName == "divide-and-conquer")
				builderFactory.reset(new DivideAndConquerFactory<Scalar>());
			
			else KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Unknown kernel-evd builder type [%s]", %kevdName);
			
			builderFactory->setSpace(our_dynamic_cast<const SpaceBase<Scalar>>(space));
			builderFactory->configure(context, json);
			return builderFactory;
		}
		
		
		
		static SelectorPtr getSelector(const std::string &context, picojson::value &json) {
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
					Index maxRank = getNumeric<Index>(contextSel, jsonSel, "max");
					Index resetRank = getNumeric<Index>(contextSel, jsonSel, "reset");
					chain->add(SelectorPtr(new RankSelector<Real,true>(maxRank, resetRank)));
				} else if (name == "ratio") {
					typename RatioSelector<Real>::AggregatorPtr aggregator(new Mean<Real>());
					Real ratio = (Real)get<double>(contextSel, jsonSel, "ratio", Eigen::NumTraits<Real>::epsilon());
					chain->add(SelectorPtr(new RatioSelector<Real>(ratio, aggregator)));
				} else KQP_THROW_EXCEPTION_F(kqp::illegal_argument_exception, "Unknown selector [%s] in [%s]", %name %context);
			}
			
			return chain;
		}
		
		static boost::shared_ptr<Cleaner<Scalar>> getCleaner(const std::string &context, picojson::value &json) override {
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
					
					Index maxRank = getNumeric<Index>(contextCleaner, jsonCleaner, "max-rank", -1);
					if (maxRank < 0)
						maxRank = std::numeric_limits<Index>::max();
					Index resetRank = getNumeric<Index>(contextCleaner, jsonCleaner, "reset-rank", 0);
					cleaner->setRankRange(resetRank, maxRank);
					
					list->add(cleaner);
				} else if (name == "qp") {
					boost::shared_ptr<CleanerQP<Real>> cleaner(new CleanerQP<Scalar>());
					
					double maxRatio = get<double>(contextCleaner, jsonCleaner, "max-ratio",  std::numeric_limits<Real>::infinity());
					double resetRatio = get<double>(contextCleaner, jsonCleaner, "reset-ratio", 0);
					cleaner->setPreImagesPerRank(resetRatio, maxRatio);
					
					Index maxRank = getNumeric<Index>(contextCleaner, jsonCleaner, "max-rank", -1);
					if (maxRank < 0)
						maxRank = std::numeric_limits<Index>::max();
					Index resetRank = getNumeric<Index>(contextCleaner, jsonCleaner, "reset-rank", 0);
					cleaner->setRankRange(resetRank, maxRank);
					
					list->add(cleaner);
				} else KQP_THROW_EXCEPTION_F(kqp::illegal_argument_exception, "Unknown cleaner [%s] in [%s]", %name %context);
				
			}
			
			return list;
		}
		
	};
	
	inline typename BuilderFactoryBase::Ptr BuilderFactoryBase::getFactory(const boost::shared_ptr<AbstractSpace> &space, picojson::value &json) {
		std::string scalarName = get<std::string>("", json, "scalar", "double");
		if (scalarName == "double") return Ptr(BuilderFactory<double>::getFactory(space, json));
		KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Unknown scalar type [%s]", %scalarName);
	}

	inline typename boost::shared_ptr<CleanerBase> BuilderFactoryBase::getCleaner(picojson::value &json, const std::string &context) {
		std::string scalarName = get<std::string>("", json, "scalar", "double");
		if (scalarName == "double") return boost::shared_ptr<CleanerBase>(BuilderFactory<double>::getCleaner(context, json.get<picojson::object>()["list"]));
		KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Unknown scalar type [%s]", %scalarName);
	}

	
	
	template<typename Scalar>
	class DirectFactory : public BuilderFactory<Scalar> {
	public:
		KQP_SCALAR_TYPEDEFS(Scalar);
		virtual ~DirectFactory() {}
		virtual boost::shared_ptr<KernelEVDBase> getBuilder() override {
			return boost::shared_ptr<KernelEVDBase>(new DenseDirectBuilder<Scalar>(this->m_space->dimension()));
		};
		virtual std::string getName() const { return "direct"; }
	};
	
	template<typename Scalar>
	class AccumulatorFactory : public BuilderFactory<Scalar> {
	public:
		KQP_SCALAR_TYPEDEFS(Scalar);
		virtual ~AccumulatorFactory() {}
		
		virtual boost::shared_ptr<KernelEVDBase> getBuilder() {
			if (this->m_space->canLinearlyCombine())
				return boost::shared_ptr<KernelEVD<Scalar>>(new AccumulatorKernelEVD<Scalar,true>(this->m_space));
			
			return boost::shared_ptr<KernelEVDBase>(new AccumulatorKernelEVD<Scalar,false>(this->m_space));
		}
		
		virtual std::string getName() const {
			return "accumulator";
		}
	};
	
	
	template<typename Scalar>
	class IncrementalFactory : public BuilderFactory<Scalar> {
	public:
		KQP_SCALAR_TYPEDEFS(Scalar);
		boost::shared_ptr<Selector<Real>> m_selector;
		float targetPreImageRatio, maxPreImageRatio;
		
		IncrementalFactory() :
		targetPreImageRatio(std::numeric_limits<float>::infinity()),
		maxPreImageRatio(std::numeric_limits<float>::infinity())
		{}
		
		virtual ~IncrementalFactory() {}
		
		virtual std::string getName() const { return "incremental"; }
		
		
		virtual void configure(const std::string &context, picojson::object &json) override {
			BuilderFactory<Scalar>::configure(context, json);
			m_selector = this->getSelector(context + ".selector", json["selector"]);
			
			if (!this->m_space->canLinearlyCombine()) {
				targetPreImageRatio = get<double>(context, json, "pre-images");
				maxPreImageRatio = get<double>(context, json, "max-pre-images", targetPreImageRatio);
			}
		}
		
		virtual boost::shared_ptr<KernelEVDBase> getBuilder() {
			IncrementalKernelEVD<Scalar> * builder  = new IncrementalKernelEVD<Scalar>(this->m_space);
			builder->setSelector(m_selector);
			builder->setPreImagesPerRank(this->targetPreImageRatio, this->maxPreImageRatio);
			
			return boost::shared_ptr<KernelEVD<Scalar>>(builder);
		}
		
	};
	
	template<typename Scalar> struct BuilderChooser;
	
	
	/** Configurator for "Divide and Conquer" KEVD */
	template<typename Scalar>
	class DivideAndConquerFactory : public BuilderFactory<Scalar> {
	public:
		KQP_SCALAR_TYPEDEFS(Scalar);
		typedef boost::shared_ptr<KernelEVD<Scalar>> KEVDPtr;
		typedef boost::shared_ptr<Cleaner<Real>> CleanerPtr;
		
		typedef boost::shared_ptr<BuilderFactory<Scalar>> BuilderPtr;
		BuilderPtr builder, merger;
		CleanerPtr builderCleaner, mergerCleaner;
		
		Index batchSize;
		
		virtual std::string getName() const { return "divide-and-conquer"; }
		
		DivideAndConquerFactory() : batchSize(100) {}
		virtual ~DivideAndConquerFactory() {}
		
		virtual void configure(const std::string &context, picojson::object &json) override {
			BuilderFactory<Scalar>::configure(context, json);
			
			batchSize = getNumeric<int>(context, json, "batch", batchSize);
			
			builder = BuilderFactory<Scalar>::getFactory(this->m_space, json["builder"], context + ".builder");
			merger = BuilderFactory<Scalar>::getFactory(this->m_space, json["merger"], context + ".merger");
			
			builderCleaner = this->getCleaner(context + ".builder.cleaner", json["builder"].get<picojson::object>()["cleaner"]);
			mergerCleaner = this->getCleaner(context + ".merger.cleaner", json["merger"].get<picojson::object>()["cleaner"]);
		}
		
		virtual boost::shared_ptr<KernelEVDBase> getBuilder() override {
			boost::shared_ptr<DivideAndConquerBuilder<Scalar>> dc(new DivideAndConquerBuilder<Scalar>(this->m_space));
			dc->setBatchSize(batchSize);
			
			
			dc->setBuilder(our_dynamic_cast<KernelEVD<Scalar>>(builder->getBuilder()));
			dc->setBuilderCleaner(builderCleaner);
			
			dc->setMerger(our_dynamic_cast<KernelEVD<Scalar>>(merger->getBuilder()));
			dc->setMergerCleaner(mergerCleaner);
			return dc;
		}
	};
	
#endif // SWIG

}


#endif