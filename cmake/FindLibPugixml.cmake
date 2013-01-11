# - Try to find ImagePUGIXML
# Once done, this will define
#
#  PUGIXML_FOUND - system has PUGIXML
#  PUGIXML_INCLUDE_DIRS - the PUGIXML include directories
#  PUGIXML_LIBRARIES - link these to use PUGIXML

include(LibFindMacros)

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(PUGIXML_PKGCONF pugixml)

# Include dir
find_path(PUGIXML_INCLUDE_DIR
  NAMES pugixml.hpp
  PATHS ${PUGIXML_PKGCONF_INCLUDE_DIRS}
)

# Finally the library itself
find_library(PUGIXML_LIBRARY
  NAMES pugixml
  PATHS ${PUGIXML_PKGCONF_LIBRARY_DIRS}
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(PUGIXML_PROCESS_INCLUDES PUGIXML_INCLUDE_DIR)
set(PUGIXML_PROCESS_LIBS PUGIXML_LIBRARY)
libfind_process(PUGIXML)
