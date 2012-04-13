#!/usr/bin/python

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