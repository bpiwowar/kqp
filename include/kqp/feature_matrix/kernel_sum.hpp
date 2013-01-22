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
#ifndef __KQP_KERNEL_SUM_H__
#define __KQP_KERNEL_SUM_H__

#include <vector>
#include <boost/lexical_cast.hpp>
#include <kqp/feature_matrix.hpp>
#include <kqp/space_factory.hpp>

namespace kqp {
	template<typename Scalar> class KernelSumSpace;

	/**
	 * Linear combination kernel
     *
     * Coefficients are squared (always positive) and normalized so they sum to 1
     *
	 */
	template <typename Scalar> 
    class KernelSumMatrix : public FeatureMatrixBase<Scalar> {
    public:       
        KQP_SCALAR_TYPEDEFS(Scalar);
        typedef KernelSumMatrix<Scalar> Self;

        KernelSumMatrix(const KernelSumSpace<Scalar> &space) {
        	for(size_t i = 0; i < space.m_spaces.size(); i++) {
        		m_matrices.push_back(space.m_spaces[i]->newMatrix());
        	}
        }

        KernelSumMatrix(const Self &other) : m_matrices(other.m_matrices) {
        }

        KernelSumMatrix(const std::vector<FMatrix> &matrices) : m_matrices(matrices) {
        }

#       ifndef SWIG
        KernelSumMatrix(Self &&other) {
			m_matrices.swap(other.m_matrices);       	
		}        
		Self &operator=(Self &&other) {
			m_matrices.swap(other.m_matrices);
			return *this;
		}
#       endif

        const FMatrix &operator[](size_t i) const {
        	return m_matrices[i];
        }

		virtual Index size() const override {
			return m_matrices.empty() ? 0 : m_matrices[0]->size();
		}

		virtual void add(const FMatrixBase &f, const std::vector<bool> *which = NULL) override {
			const Self &other = f.template as<Self>();
			for(size_t i = 0; i < m_matrices.size(); i++)
				m_matrices[i]->add(other[i], which);
		} 

        virtual FMatrixBasePtr subset(const std::vector<bool>::const_iterator &begin, 
        	const std::vector<bool>::const_iterator &end) const override {
        	boost::shared_ptr<Self> matrix = boost::shared_ptr<Self>(new Self());

			for(size_t i = 0; i < m_matrices.size(); i++) {
				matrix->m_matrices.push_back(FMatrix(m_matrices[i]->subset(begin, end)));
			}
			return matrix;
        }

        
        /** Assignement */
        virtual FMatrixBase &operator=(const FeatureMatrixBase<Scalar> &other) override {
        	this->m_matrices = other.template as<Self>().m_matrices;
        	return *this;
        }
        
        /** Copy */
        virtual FMatrixBasePtr copy() const override {
        	return FMatrixBasePtr(new Self(*this));
        }


        private:
        	KernelSumMatrix() {}
        	std::vector<FMatrix> m_matrices;
    };

    /**
     * @brief A feature matrix where vectors are sparse vectors in a high dimensional space
     *
     * This class makes the hypothesis that vectors have only a few non null components (compared to the dimensionality of the space).
     *
     * @ingroup FeatureMatrix
     */
    template <typename Scalar> 
    class KernelSumSpace : public SpaceBase<Scalar> {
    public:
		KQP_SCALAR_TYPEDEFS(Scalar);

		typedef KernelSumMatrix<Scalar> TMatrix;
		friend class KernelSumMatrix<Scalar>;

		/** Add a new space */
		void addSpace(Real weight, const FSpace &space) {
			m_weights.push_back(weight);
            m_sum += weight * weight;
			m_spaces.push_back(space);
		}

        KernelSumSpace() : m_sum(0) {}
        KernelSumSpace(const KernelSumSpace &other) : m_spaces(other.m_spaces),  m_weights(other.m_weights), m_sum(0)  {}
        ~KernelSumSpace() {}

        static FSpace create() { return FSpace(new KernelSumSpace()); }

        virtual Index dimension() const override { 
        	Index n = 1;
        	for(size_t i = 0; i < m_spaces.size(); i++) {
        		n *= m_spaces[i]->dimension();
        		if (n < 0) return -1;
        	}

        	return n;
        }

        virtual boost::shared_ptr< SpaceBase<Scalar> > copy() const {
            return FSpacePtr(new KernelSumSpace());
        }

        virtual bool _canLinearlyCombine() const override { return false; }

        virtual FMatrixBasePtr newMatrix() const override { 
        	return FMatrixBasePtr(new TMatrix(*this)); 
        }
        virtual FMatrixBasePtr newMatrix(const FeatureMatrixBase<Scalar> &other) const override { 
        	return FMatrixBasePtr(new TMatrix(other.template as<TMatrix>())); 
        }        


        virtual const ScalarMatrix &k(const FMatrixBase &mX) const {
            // FIXME: Big memory leak here (the feature space should be notified when a feature matrix is deleted) !!!
            // So we suppose we are not multithreaded... and compute it each time.
            // ScalarMatrix &gram = m_gramCache[static_cast<const void*>(&mX)];
            
            static ScalarMatrix gram;

            gram = ScalarMatrix::Zero(mX.size(), mX.size());

            for(size_t i = 0; i < m_spaces.size(); i++) {
                gram += getNormalizedWeight(i) * m_spaces[i]->k(mX.template as<TMatrix>()[i]);
            }    
            return gram;
        }

        virtual ScalarMatrix k(const FMatrixBase &mX1, const ScalarAltMatrix &mY1, const RealAltVector &mD1, 
                               const FMatrixBase &mX2, const ScalarAltMatrix &mY2, const RealAltVector &mD2) const override {
        	ScalarMatrix k = ScalarMatrix::Zero(mX1.size(), mX2.size());

        	for(size_t i = 0; i < m_spaces.size(); i++) {
        		k += getNormalizedWeight(i) * m_spaces[i]->k(mX1.template as<TMatrix>()[i], mY1, mD1, mX2.template as<TMatrix>()[i], mY2, mD2);
        	}

        	return k;
        }

     
        
        virtual void update(std::vector< KernelValues<Scalar> > &values, int kOffset = 0) const override {
            auto &self = values[kOffset];
            kOffset++;

            self._inner = self._innerX = self._innerY = 0;
            for(size_t i = 0; i < m_spaces.size(); i++) {
                m_spaces[i]->update(values, kOffset);
                const auto &child = values[kOffset];
                self._inner += getNormalizedWeight(i) * child._inner;
                self._innerX += getNormalizedWeight(i) * child._innerX;
                self._innerY += getNormalizedWeight(i) * child._innerY;
                kOffset += m_spaces[i]->numberOfKernelValues();
            }

        }

        virtual void updatePartials(Real alpha, std::vector<Real> &partials, int offset, const std::vector< KernelValues<Scalar> > &values,  int kOffset, int mode) const override {
            Scalar k = values[kOffset].inner(mode);
            kOffset += 1;

            for(size_t i = 0; i < m_spaces.size(); i++) {
                partials[offset] += alpha * 2. * (m_weights[i]*m_weights[i]) * (values[kOffset].inner(mode) - k) / m_sum;
                m_spaces[i]->updatePartials(alpha * getNormalizedWeight(i), partials, offset + 1, values, kOffset, mode);

                offset += m_spaces[i]->numberOfParameters() + 1;
                kOffset +=  m_spaces[i]->numberOfKernelValues();
            }

        }

        virtual int numberOfKernelValues() const {
            int n = 1;
            for(size_t i = 0; i < m_spaces.size(); i++)
                n += m_spaces[i]->numberOfKernelValues();
            return n;
        }

        virtual int numberOfParameters() const {
            int n = m_spaces.size();
        	for(size_t i = 0; i < m_spaces.size(); i++)
        		n += m_spaces[i]->numberOfParameters();
        	return n;
        }

        virtual void getParameters(std::vector<Real> & parameters, int offset) const {
        	for(size_t i = 0; i < m_spaces.size(); i++) {
				parameters[offset] = m_weights[i];
                m_spaces[i]->getParameters(parameters, offset+1);
                offset += m_spaces[i]->numberOfParameters() + 1;
			}
        }

        virtual void setParameters(const std::vector<Real> & parameters, int offset)  {
            m_sum = 0;
        	for(size_t i = 0; i < m_spaces.size(); i++) {
				m_weights[i] = parameters[offset];
                m_sum += m_weights[i];
                m_spaces[i]->setParameters(parameters, offset + 1);
                offset += m_spaces[i]->numberOfParameters() + 1;
			}
		}

        
        static const std::string &name() { static std::string NAME("sum"); return NAME; }

        FSpacePtr space(size_t i) { return m_spaces[i]; }
        FSpaceCPtr space(size_t i) const { return m_spaces[i]; }
        Real weight(size_t i) const { return m_weights[i]; }
        size_t size() const { return m_spaces.size(); }

        virtual void load(const pugi::xml_node &node) {
            static const std::string SUB_NAME("sub");

            m_sum = 0;
            m_spaces.clear();
            m_weights.clear();

            for(auto child: node) {
                if (child.type() == pugi::xml_node_type::node_element && child.name() == SUB_NAME) {
                    Real weight = kqp::attribute<Real>(child, "weight", 1.);

                    pugi::xml_node selected;
                    for(auto grandchild: child) {
                        if (grandchild.type() == pugi::xml_node_type::node_element) {
                            if (selected.empty()) {
                                selected = grandchild;
                            } else { 
                                KQP_THROW_EXCEPTION(exception, "A unary kernel element should have no more than one child"); 
                            }
                        }
                    }
                    if (selected.empty())
                        KQP_THROW_EXCEPTION(exception, "A unary kernel element should have one child");

                    auto space = kqp::our_dynamic_cast< SpaceBase<Scalar> >(SpaceFactory::load(selected));

                    addSpace(weight, space);
                }
            }
        }

        virtual pugi::xml_node save(pugi::xml_node &node) const override {
            static const std::string SUB_NAME("sub");
            pugi::xml_node self = SpaceBase<Scalar>::save(node);
            for(size_t i = 0; i < m_spaces.size(); i++) {
                auto child = self.append_child(SUB_NAME.c_str());
                child.append_attribute("weight") = boost::lexical_cast<std::string>(m_weights[i]).c_str();
                m_spaces[i]->save(child);
            }
        }

        inline Real getNormalizedWeight(size_t i) const {
            return m_weights[i] * m_weights[i] / m_sum;
        }
    private:
    	std::vector<FSpace> m_spaces;
    	std::vector<Real> m_weights;
        mutable Real m_sum;
	};

}


#endif