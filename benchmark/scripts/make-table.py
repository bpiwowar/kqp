#!/usr/bin/python

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
 

import sys

keys = {}
tableKeys = []
table = []

for path in sys.argv[1:]:
    row = [ "na" ] * len(keys)
    
    for line in file(path):
        fields = line.strip().split("\t")
        if len(fields) != 2: 
            sys.stderr.write("Skipping line\n")
            continue
        [key, value] = fields
        
        idKey = keys.get(key, -1)
        if idKey != -1: 
            row[idKey] = value
        else:
            idKey = len(keys)
            keys[key] = idKey
            tableKeys.append(key)
            row.append(value)
    table.append(row)

print "\t".join(tableKeys)
for row in table:
    row += ["na"] * (len(row) - len(keys))
    print "\t".join(row)