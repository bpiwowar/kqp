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
    """Add a prefix to each option"""
    m = {}
    for k, v in d.iteritems():
        m[prefix + k] = v
    return m

# DIM UPDATES RANKRESET max-rank PREIMRESET max_pre_images


def problems():
    yield {"dimension": 100, "updates": 100, "no-lc": False, "rank": 100, "max_rank": 100, "pre_images": 1, "max_pre_images": 1}
    yield {"dimension": 100, "updates": 200, "no-lc": True, "rank": 80, "max_rank": 120, "pre_images": 1.5, "max_pre_images": 2}
    yield {"dimension": 1000, "updates": 1000, "no-lc": True, "rank": 80, "max_rank": 120, "pre_images": 1.5, "max_pre_images": 2}
    yield {"dimension": 10000, "updates": 10000, "no-lc": True, "rank": 80, "max_rank": 120, "pre_images": 1.5, "max_pre_images": 2}
    yield {"dimension": 10000, "updates": 10000, "no-lc": False, "rank": 80, "max_rank": 120, "pre_images": 1.5, "max_pre_images": 2}

# EVD classes define a process method that
# 1. Takes the problem definition + builder parameters in input
# 2. returns a couple (name, parameters) or None if the problem is not tackable


def union(*dicts):
    return dict(sum(map(lambda dct: list(dct.items()), dicts), []))


def remove(keys, map):
    r = {}
    for key, value in map.items():
        if key not in keys:
            r[key] = value
    return r


def select(keys, map):
    r = {}
    for key in keys:
        if key in map:
            r[key] = map[key]
    return r


class Base:
    def process(self, problem):
        return addPrefix("-", select(["rank", "pre_images", "max_rank", "max_pre_images"], problem))


class Direct(Base):
    def process(self, problem):
        if problem["dimension"] > 1000:
            return None
        if problem["no-lc"]:
            return None
        return union({"": "direct"}, Base.process(self, problem))


class Accumulator(Base):
    def process(self, problem):
        # Skip to big problems
        if problem["updates"] > 1000:
            return None
        return union({"": "accumulator"}, Base.process(self, problem))


class DivideAndConquer(Base):
    def __init__(self):
        self.builder = Accumulator()
        self.merger = Accumulator()

    def process(self, problem):
        b = addPrefix("-builder", self.builder.process(remove(["rank", "pre_images", "max_rank", "max_pre_images"], problem)))
        m = addPrefix("-merger", self.merger.process(problem))
        batch_size = problem["rank"]
        return union({"": "divide-and-conquer", "-batch_size": batch_size}, b, m, Base.process(self, problem))


def builders():
    return [Direct(), Accumulator(), DivideAndConquer()]

# === Main loop ===


def commandLine(d):
    args = []
    for k, v in sorted(d.iteritems()):
        args.append("--" + k)

        if v is None or (type(v) == list and len(v) == 0):
            pass
        elif type(v) == list:
            args = args + ["%s" % x for x in v]
        else:
            args.append("%s" % v)
    return args

#
for problem in problems():
    for builder in builders():

    # Process the problem by the builder
        args = builder.process(problem)
        if args is None:
            continue
        args = addPrefix("builder", args)

        # Build the command line
        mArgs = [bmPath, "kernel-evd", "--dimension", str(problem["dimension"]), "--updates", str(problem["updates"])]
        if problem["no-lc"]:
            mArgs += ["--no-lc"]
        mArgs += commandLine(args)
        key = hashlib.sha512("@@".join(mArgs)).hexdigest()

        outpath = os.path.join(outdir, key + ".dat")

        if (os.path.exists(outpath)):
            print "[Skipping %s/%s]" % (outpath, args["builder"])
        else:
            print "* Running with %s" % " ".join(mArgs)
            p = Popen(mArgs, stdout=file("%s.out" % outpath, "w"), stderr=file("%s.err" % outpath, "w"))
            p.wait()
            if p.returncode == 0:
                os.rename("%s.out" % outpath, outpath)
            else:
                sys.stderr.write("[Failure]\nCheck file %s\n" % ("%s.err" % outpath))
