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

#include <kqp/kqp.hpp>
#include <kqp/kernel_evd/factory.hpp>

DEFINE_LOGGER(logger,  "kqp.benchmark.kernel-evd");

namespace kqp {	
	template<typename Scalar>
	struct KernelEVDBenchmark {
		KQP_SCALAR_TYPEDEFS(Scalar);
		
		typedef boost::shared_ptr<Cleaner<Scalar>> CleanerPtr;
		
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
		
		boost::hellekalek1995 generator;
		boost::uniform_01<double> uniformGenerator;
		
		ScalarMatrix m_genVectors;
		CleanerPtr m_cleaner;
		
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
		
		
		
		//! Initialise
		void init() {
			generator.seed(seed);
			
			if (nbVectors > 0) {
				KQP_LOG_INFO_F(logger, "Creating %d base vectors in dimension %d", %nbVectors %dimension);
				m_genVectors = ScalarMatrix::Random(dimension, nbVectors);
			}
		}
		
		//! Get the next feature matrix + mixture matrix
		void getNext(Real &alpha, ScalarMatrix &m, ScalarMatrix &mA) const {
			alpha = Eigen::internal::abs(Eigen::internal::random_impl<Real>::run()) + 1e-3;
			
			int k = (int)std::abs(Eigen::internal::random_impl<double>::run() * (double)(max_preimages-min_preimages)) + min_preimages;
			int p = (int)std::abs(Eigen::internal::random_impl<double>::run() * (double)(max_lc-min_lc)) + min_lc;
			KQP_LOG_DEBUG(logger, boost::format("Pre-images (%dx%d) and linear combination (%dx%d)") % dimension % k % k % p);
			
			// Generate the linear combination matrix
			mA = ScalarMatrix::Random(k, p);
			
			// Generate k pre-images
			if (nbVectors > 0) {
				// TODO: weights should be generated according to some distribution
				RealVector weights = RealVector::Random(nbVectors);
				
				m = m_genVectors * weights.asDiagonal() * ScalarMatrix::Random(nbVectors,k) + noise * ScalarMatrix::Random(dimension, k);
			} else {
				m = ScalarMatrix::Random(dimension, k);
			}
		}
		
		
		int run(picojson::object &value) override {
			std::string name = get<std::string>("", value, "name", "na");
			
			seed = get<double>("", value, "seed", seed);
			useLC = get<bool>("", value, "lc", useLC);
			noise = get<double>("", value, "noise", noise);
			nbVectors = getNumeric<int>("", value, "nb-vectors", nbVectors);
			
			
			dimension = getNumeric<int>("", value, "dimension", dimension);
			updates = getNumeric<int>("", value, "updates", updates);
			
			
			FSpace fs = DenseSpace<Scalar>::create(dimension);
			fs->setUseLinearCombination(useLC);
			init();
			
			BuilderFactoryOptions options = { useLC, dimension };
			
			boost::shared_ptr<BuilderFactory<Scalar>> factory = BuilderFactory<Scalar>::getBuilder(options, value["builder"]);
			
			if (value.find("cleaner") != value.end())
				m_cleaner = BuilderFactory<Scalar>::getCleaner("", value["cleaner"]);
			
			boost::shared_ptr<KernelEVD<Scalar>> builder = factory->getBuilder(fs, options);
			
			std::clock_t total_time = 0;
			
			// --- Computing Kernel EVD
			
			KQP_LOG_INFO_F(logger, "Computing kernel EVD with builder %s", % KQP_DEMANGLE(*builder));
			
			std::clock_t c_start = std::clock();
			
			std::srand(seed);
			ScalarMatrix mU, mA;
			Real alpha;
			for(int i = 0; i < updates; i++) {
				getNext(alpha, mU, mA);
				builder->add(alpha, Dense<Scalar>::create(mU), mA);
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
			
			std::srand(seed);
			for(int i = 0; i < updates; i++) {
				// Get the next matrix
				getNext(alpha, mU, mA);
				
				// Project
				ScalarMatrix mW = mYTXT * mU * mA;
				sumWWT.template selfadjointView<Eigen::Lower>().rankUpdate(mW, alpha);
				orthogononal_error += Eigen::internal::abs(alpha) * (mA.adjoint() * mU.adjoint() * mU * mA - mW.adjoint() * mW).squaredNorm();
			}
			
			double dnorm = ScalarMatrix(result.mD.asDiagonal()).squaredNorm();
			double s_error = (ScalarMatrix(sumWWT.template selfadjointView<Eigen::Lower>()) - ScalarMatrix(result.mD.asDiagonal())).squaredNorm();
			
			errors["o_error"] = picojson::value(orthogononal_error);
			errors["s_error"] = picojson::value(s_error);
			
			errors["o_error_rel"] = picojson::value(orthogononal_error / dnorm);
			errors["s_error_rel"] = picojson::value(s_error / dnorm);
			
			errors["pre_images"] = picojson::value((double)result.mX->size());
			errors["rank"] = picojson::value((double)result.mY.cols());
			
			
			return 0;
			
		}
		
	};
	
	
	
	template<typename Scalar> int  run_kevd_benchmark(picojson::value &v) {
		KernelEVDBenchmark<Scalar> kernelEVD;
		return kernelEVD.run(v.get<picojson::object>());
	}
	
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
		
		std::string scalarName = get<std::string>("", v, "scalar", "double");
		int code;
		
		if (scalarName == "double") code = run_kevd_benchmark<double>(v);
		else KQP_THROW_EXCEPTION_F(illegal_argument_exception, "Unknown scalar type [%s]", %scalarName);
		
		
		v.serialize(std::ostream_iterator<char>(std::cout));
		std::cout << std::endl;
		
		return code;
	}
} // kqp


