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


out = sys.stdout

times = ['kevd', 'orthonormalizing', 'cleaning', 'total']
errors = [u'rank', u'pre_images', u's_error', u'o_error']

out.write("""<html>
    <head>
    <title>KQP benchmark</title>
    <script type="text/javascript" src="http://ajax.googleapis.com/ajax/libs/jquery/1.8/jquery.min.js"></script>
    <script type="text/javascript">
$(document).ready(function() {
});""")


out.write("""</script>
</head>
<body>
<h1>KQP benchmark results</h1>
""")


def json2html(out, r):
    if type(r) == dict:
        out.write("<dl>")
        for k, v in r.iteritems():
            out.write("<dt>%s</dt><dd>" % k)
            json2html(out, v)
            out.write("</dd>")
        out.write("</dl>\n")
    elif type(r) == list:
        out.write("<ul>")
        for v in r:
            out.write("<li>")
            json2html(out, v)
            out.write("</li>")
        out.write("</ul>\n")
    else:
        out.write("%s" % r)

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
    out.write("<table><thead>")
    for t in times:
        out.write("<th>%s</th>" % t)
    for e in errors:
        out.write("<th>%s</th>" % e)
    out.write("<th>Description</th>")
    out.write("</thead>")

    for r in v:
        out.write("<tr>")
        for t in times:
            out.write("<td>%s</td>" % r["time"][t])
        for e in errors:
            out.write("<td>%s</td>" % r["error"][e])

        out.write("<td>")
        json2html(out, r)
        out.write("</td></tr>")
    out.write("</table>")

out.write("""
<body></html>""")
