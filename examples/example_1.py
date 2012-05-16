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
# along with KQP.  If not, see <http:#www.gnu.org/licenses/>.

from kqp import *

dim = 10

# Feature space
fs = DenseSpaceDouble.create(10)

# Creating an incremental builder
print("Creating a KEVD builder\n")
kevd = KEVDAccumulatorDouble(fs)

# Add 10 vectors with $\alpha_i=1$
print("Adding 10 vectors\n")
for i in range(10):
    # Adds a random $\varphi_i$
    m = EigenMatrixDouble(dim,1)
    m.randomize()
    kevd.add(DenseDouble.create(m))

# Get the result $\rho \approx X Y D Y^\dagger X^\dagger$
print "Getting the result"
d = kevd.getDecomposition()

# --- Compute a kEVD for a subspace

print("Creating a KEVD builder (event)\n")

kevd_event = KEVDAccumulatorDouble(fs)

for i in range(3):
    # Adds a random $\varphi_i$
    m = EigenMatrixDouble(dim,1)
    m.randomize()
    kevd_event.add(DenseDouble.create(m))

# --- Compute some probabilities


# Setup densities and events
d = kevd.getDecomposition()
print "Creating the density rho and event E"
rho = DensityDouble(kevd)
rho.normalize()
event = EventDouble(kevd_event)
print "Computing some probabilities"

# Compute the probability
print "Probability = %g\n" % rho.probability(event)

# Conditional probability
rho_cond = event.project(rho).normalize()
print "Entropy of rho/E = %g\n" % rho_cond.entropy()

# Conditional probability (orthogonal event)
rho_cond_orth = event.project(rho, true).normalize()
print "Entropy of rho/not E = %g\n" % rho_cond.entropy()

