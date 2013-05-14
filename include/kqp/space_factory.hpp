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


#ifndef __KQP_SPACE_FACTORY_H__
#define __KQP_SPACE_FACTORY_H__

#include <fstream>
#include <sstream>

#include <kqp/picojson.hpp>
#include <kqp/feature_matrix.hpp>
#include <kqp/cleanup.hpp>

namespace kqp {
    template<typename Scalar, typename SpaceInstance> static AbstractSpace* CONSTRUCTOR() {
		return new SpaceInstance();
    };
	
    //! Factory for
    class SpaceFactory {
    public:
        //! Constructor type for spaces
        typedef AbstractSpace * (*BaseConstructor)();
		
        static inline void registerSpace(const std::string &name, BaseConstructor constructor) {
			constructors()[name] = constructor;
        }
		
        //! Load a space, starting with an XML node
        static inline boost::shared_ptr<AbstractSpace> load(const picojson::object &value) {
			std::string scalar_name = get<std::string>("", value, "scalar", "double");
			std::string name = get<std::string>("", value, "name") + "[" + scalar_name + "]";
			BaseConstructor constructor = constructors()[name];
			if (!constructor)
				KQP_THROW_EXCEPTION_F(exception, "Cannot retrieve space with name %s and scalar %s", %name %scalar_name);
			boost::shared_ptr<AbstractSpace> fSpace (constructor());
			fSpace->load(value);
			return fSpace;
        }
		
        //! Load from an XML file
        static inline boost::shared_ptr<AbstractSpace> loadFromFile(const std::string &filename) {
			picojson::value v = readJsonFromFile(filename);
			return load(v.get<picojson::object>());
        }
		
        //! Load from an XML string
        static inline boost::shared_ptr<AbstractSpace> loadFromString(const std::string &jsonstring) {
			picojson::value v = readJsonFromString(jsonstring);
			return load(v.get<picojson::object>());
        }
		
		
        //! Save to a file
        static inline void saveToFile(const std::string &filename, AbstractSpace &space) {
			picojson::object json = space.save();
			std::ofstream out(filename.c_str());
			picojson::value(json).serialize(std::ostream_iterator<char>(out));
        }
		
        //! Save to a string
        static inline std::string getJSONString(const AbstractSpace &space) {
			picojson::object json = space.save();
			std::ostringstream out;
			picojson::value(json).serialize(std::ostream_iterator<char>(out));
			return out.str();
        }		
		
#ifndef SWIG
        //! Registration class
        template<typename Scalar, class SpaceInstance> struct Register {
			Register() {
				std::string name = SpaceInstance::NAME() + "[" + ScalarInfo<Scalar>::name() + "]";
				SpaceFactory::registerSpace(name, CONSTRUCTOR<Scalar, SpaceInstance>);
			}
        };
        static std::vector<std::string> registered() {
			std::vector<std::string> list;
			for(auto e: constructors())
				if (e.second)
					list.push_back(e.first);
				else
					list.push_back("[UNDEFINED] " + e.first);
			return list;
        }
		
		//! Map from names to constructors
		static std::map<std::string, BaseConstructor> &constructors() {
            static std::map<std::string, BaseConstructor> _CONSTRUCTORS_;
            return _CONSTRUCTORS_;
		}
#endif		
		
    };
}
#endif
