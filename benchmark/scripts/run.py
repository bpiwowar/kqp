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
import json

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
    yield {"name": "small (LC)", "dimension": 100, "updates": 100, "lc": False, "" "rank": 100, "max_rank": 100, "pre_images": 1, "max_pre_images": 1.0}
    yield {"name": "small (K)", "dimension": 100, "updates": 200, "lc": True, "rank": 80, "max_rank": 120, "pre_images": 1.5, "max_pre_images": 2.0}
    yield {"name": "medium (LC)", "dimension": 1000, "updates": 1000, "lc": True, "rank": 80, "max_rank": 120, "pre_images": 1.5, "max_pre_images": 2.0}
    yield {"name": "medium (K)", "dimension": 1000, "updates": 1000, "lc": False, "rank": 80, "max_rank": 120, "pre_images": 1.5, "max_pre_images": 2.0}
    yield {"name": "big (LC)", "dimension": 10000, "updates": 10000, "lc": True, "rank": 80, "max_rank": 120, "pre_images": 1.5, "max_pre_images": 2.0}
    yield {"name": "big (K)", "dimension": 10000, "updates": 10000, "lc": False, "rank": 80, "max_rank": 120, "pre_images": 1.5, "max_pre_images": 2.0}

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


def cleaner(problem, imageCleaner):
    c = [
        {
            "name": "rank",
            "selector": [
                    {
                        "name": "rank",
                        "max": problem["max_rank"],
                        "reset": problem["rank"]
                    }
            ]
        },
        {
            "name": "unused"
        },
        imageCleaner(problem)
    ]
    return c


def qpImageCleaner(problem):
    return {
        "name": "qp",
        "max": problem["max_pre_images"],
        "reset": problem["pre_images"]
    }


def nullImageCleaner(problem):
    return {
        "name": "null",
        "max": problem["max_pre_images"],
    }


def select(keys, map):
    r = {}
    for key in keys:
        if key in map:
            r[key] = map[key]
    return r


class Base:
    def process(self, problem):
        p = select(["name", "dimension", "updates", "lc"], problem)
        p["cleaner"] = cleaner(problem, qpImageCleaner)
        return p


class Direct(Base):
    def process(self, problem):
        if problem["dimension"] > 1000:
            return None
        if not problem["lc"]:
            return None
        return union({"builder": {"name": "direct"}}, Base.process(self, problem))


class Accumulator(Base):
    def process(self, problem):
        # Skip problems with large number of updates
        if problem["updates"] > 1000:
            return None
        return union({"builder": {"name": "accumulator"}}, Base.process(self, problem))


class DivideAndConquer(Base):
    def __init__(self, imageCleaner):
        self.builder = Accumulator()
        self.merger = Accumulator()
        self.imageCleaner = imageCleaner

    def process(self, problem):
        builder = {"name": "divide-and-conquer", "batch": problem["rank"]}
        builder["builder"] = {"name": "accumulator"}
        builder["merger"] = {"name": "accumulator", "cleaner": cleaner(problem, self.imageCleaner)}
        model = {"builder": builder}
        return union(model, Base.process(self, problem))


def builders():
    return [Direct(), Accumulator(), DivideAndConquer(qpImageCleaner), DivideAndConquer(nullImageCleaner)]

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

        # Build the command line
        jsonString = json.dumps(args)
        key = hashlib.sha512("@@".join(jsonString)).hexdigest()
        print jsonString

        outpath = os.path.join(outdir, key + ".dat")
        jsonpath = os.path.join(outdir, key + ".json")
        with open(jsonpath, "w") as text_file:
            text_file.write(jsonString)

        mArgs = [bmPath, "kernel-evd", jsonpath]

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

        print
