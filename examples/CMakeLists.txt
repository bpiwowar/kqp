# This file is part of the Kernel Quantum Probability library (KQP).
# 
# KQP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#  
# KQP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#  
# You should have received a copy of the GNU General Public License
# along with KQP.  If not, see <http://www.gnu.org/licenses/>.

INCLUDE_DIRECTORIES(${kqp_SOURCE_DIR}/src)
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

ADD_EXECUTABLE(example_1 example_1.cpp)
TARGET_LINK_LIBRARIES(example_1 kqp)

ADD_DEPENDENCIES(run-tests DEPENDS example_1)

ADD_TEST(example_1 example_1)
