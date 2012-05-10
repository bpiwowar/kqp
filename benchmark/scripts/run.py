#!/bin/python

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
from subprocess import Popen
import hashlib
import os.path

# === Argument checking ===
if len(sys.argv) != 3:
  sys.stderr.write("Expected two arguments: <benchmark executable path> <output directory>\n")
  sys.exit(1)

bmPath = sys.argv[1]
outdir = sys.argv[2]

# === Generation of benchmarks ===

def addPrefix(prefix, d):
  m = {}
  for k,v in d.iteritems():
    m[prefix + k] = v
  return m

# DIM UPDATES RANKRESET RANKMAX PREIMRESET PREIMMAX
def problems1():
  yield { "problem": {"dimension": 100, "updates": 100}, "builder": {"rank": 80, "rankMax": 120, "preIm": 1.2, "preImMax":  2} }
  yield { "problem": {"dimension": 100, "updates": 100, "no-lc": []}, "builder": {"rank": 80, "rankMax": 120, "preIm": 1.2, "preImMax":  2} }

def baseKEVD(name, g):
  for m in g:
    x = m["problem"]
    x["builder"] = name
    x.update(addPrefix("builder-", { "rank": m["builder"]["rank"] }))
    if (x.has_key("no-lc")):
      x.update(addPrefix("builder-", { }))    
    yield [x, m]

def direct(g):
  for m in baseKEVD("direct", g):
    if (m[0].has_key("no-lc")): continue
    yield m[0]
    
def accumulator(g):
  for m in baseKEVD("accumulator", g): 
    print m[0]
    yield m[0]

def incremental(g):
  for r in baseKEVD("incremental", g):
    x = r[0]
    b = r[1]["builder"]
    x["builder"] = "incremental"
    x.update(addPrefix("builder-", { "maxrank": b["rankMax"] }))
    if (x.has_key("no-lc")):
      x.update(addPrefix("builder-", { "maxpreimages": b["preImMax"] }))    
    yield x
  
# === Main loop ===

def commandLine(d):
  args = []
  for k,v in sorted(d.iteritems()):
    args.append("--" + k)
    
    if v is None or (type(v) == list and len(v) == 0): pass
    elif type(v) == list: args = args + ["%s" % x for x in v]
    else: args.append("%s" % v)
  return args
  
# Direct 
for method in [ direct(problems1()), accumulator(problems1()), incremental(problems1()) ]:
  for mArgs in method:
    args = [ bmPath, "kernel-evd" ] + commandLine(mArgs)
    key = hashlib.sha512("@@".join(args)).hexdigest()
    
    outpath = os.path.join(outdir, key + ".dat");
    
    if (os.path.exists(outpath)):
      print "[Skipping %s/%s]" % (outpath,mArgs["builder"])
    else:
      print "* Running with %s" % args
      p = Popen(args, stdout=file("%s.out" % outpath, "w"), stderr=file("%s.err" % outpath, "w"))
      p.wait()
      if p.returncode == 0:
        os.rename("%s.out" % outpath, outpath)
      else:
        sys.stderr.write("[Failure]\nCheck file %s\n" % ("%s.err" % outpath))
        
# With Divide and conquer
pass
