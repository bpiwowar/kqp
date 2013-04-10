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
import json
import os


out = sys.stdout
basedir = os.path.dirname(os.path.realpath(__file__))

times = ['kevd', 'orthonormalizing', 'cleaning', 'total']
errors = [u'rank', u'pre_images', u's_error', u'o_error']

out.write("""<html>
    <head>
    <title>KQP benchmark</title>
    <script type="text/javascript" src="%(basedir)s/jquery-1.9.1.min.js"></script>
    <script type="text/javascript" src="%(basedir)s/jquery.hotkeys.js"></script>
    <script type="text/javascript" src="%(basedir)s/jstree/jquery.jstree.js"></script>
    <link rel="stylesheet" href="%(basedir)s/tablesorter/themes/blue/style.css" type="text/css" id="" media="print, projection, screen" />
    <script type="text/javascript" src="%(basedir)s/tablesorter/jquery.tablesorter.min.js"></script>
    <script type="text/javascript">
$(document).ready(function() {
    $(".treeview").jstree({
        "plugins" : ["html_data", "themes", "ui", "hotkeys"]
        });
    $("table.perf").tablesorter();
});""" % {"basedir": basedir})


out.write("""</script>
</head>
<body>
<h1>KQP benchmark results</h1>
""")


def select(include, keys, map):
    r = {}
    for key in map.keys():
        if (key in keys) == include:
            r[key] = map[key]
    return r


def json2html(out, r):
    if type(r) == dict:
        out.write("<ul>")
        for k, v in r.iteritems():
            out.write("<li><a>%s</a>" % k)
            json2html(out, v)
            out.write("</li>")
        out.write("</ul>\n")
    elif type(r) == list:
        out.write("<ul>")
        i = 0
        for v in r:
            i = i + 1
            out.write("<li><a>[%d]</a>" % i)
            json2html(out, v)
            out.write("</li>")
        out.write("</ul>\n")
    else:
        out.write("<ul><li><a>%s</a></li></ul>" % r)

perfs = {}
for path in sys.argv[1:]:
    with open(path, 'r') as content_file:
        r = json.load(content_file)
        x = r["name"]
        if x in perfs:
            perfs[x].append(r)
        else:
            perfs[x] = [r]

for k, v in perfs.iteritems():
    out.write("<h2>%s</h2>" % k)
    json2html(out, select(False, ["builder", "cleaner", "name", "time", "error"], v[0]))
    out.write("<table class='perf tablesorter'><thead>")
    for t in times:
        out.write("<th>%s</th>" % t)
    for e in errors:
        out.write("<th>%s</th>" % e)
    out.write("<th>builder</th>")
    out.write("</thead>")

    for r in v:
        out.write("<tr>")
        for t in times:
            out.write("<td>%s</td>" % r["time"][t])
        for e in errors:
            out.write("<td>%s</td>" % r["error"][e])

        out.write("<td>%s</td>" % r["builder"]["name"])

        out.write("<td class='treeview'>")
        json2html(out, select(True, ["builder", "cleaner"], r))
        out.write("</td></tr>")
    out.write("</table>")

out.write("""
<body></html>""")
