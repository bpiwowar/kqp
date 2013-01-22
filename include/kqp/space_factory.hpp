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

#include "feature_matrix.hpp"

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
        static inline boost::shared_ptr<AbstractSpace> load(const pugi::xml_node &node) {
          auto scalar_node = node.attribute("scalar");
          std::string scalar_name = scalar_node.empty() ? scalar_node.value() : "double";
          std::string name = std::string(node.name()) + "[" + scalar_name + "]";
          BaseConstructor constructor = constructors()[name];
          if (!constructor)
            KQP_THROW_EXCEPTION_F(exception, "Cannot retrieve space with name %s and scalar %s / %s", %node.name() % scalar_name % name);
          boost::shared_ptr<AbstractSpace> fSpace (constructor());
          fSpace->load(node);
          return fSpace;
        }

        //! Load from an XML file
        static inline boost::shared_ptr<AbstractSpace> loadFromFile(const std::string &filename) {
          pugi::xml_document doc;
          pugi::xml_parse_result result = doc.load_file(filename.c_str());
          return load(result, doc);
        }

        //! Load from an XML string
        static inline boost::shared_ptr<AbstractSpace> loadFromString(const std::string &xmlstring) {
          pugi::xml_document doc;
          pugi::xml_parse_result result = doc.load(xmlstring.c_str());
          return load(result, doc);
        }

        //! Save to a file
        static inline void saveToFile(const std::string &filename, AbstractSpace &space) {
          pugi::xml_document doc;
          space.save(doc);
          doc.save_file(filename.c_str());
        }

        //! Save to a string
        static inline std::string getXMLString(AbstractSpace &space) {
          pugi::xml_document doc;
          space.save(doc);
          std::ostringstream buffer;
          doc.save(buffer);
          return buffer.str();
        }


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

        private:
        static inline boost::shared_ptr<AbstractSpace> load(const pugi::xml_parse_result &result, const pugi::xml_document &doc) {
          if (result) {
            pugi::xml_node node = doc.first_child();
            if (node.empty())
              KQP_THROW_EXCEPTION(exception, "XML document is empty");
            return SpaceFactory::load(node);
          } 
          
          KQP_THROW_EXCEPTION(exception, "XML document is invalid");

        }

          //! Map from names to constructors
          static std::map<std::string, BaseConstructor> &constructors() {
            static std::map<std::string, BaseConstructor> _CONSTRUCTORS_;
            return _CONSTRUCTORS_;
          }

    };
}
#endif