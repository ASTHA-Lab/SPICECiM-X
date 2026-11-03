#!/usr/bin/env python3
"""
Add a trace-matrix traffic mode to BookSim2 (github.com/booksim/booksim2).

Upstream BookSim2 only has synthetic patterns (uniform, transpose, hotspot,
...), which cannot express "the global buffer sends this layer's activations
to these particular tiles".  This patch adds, without touching anything else:

    matrix_file = "<path>";        (new string option)
    traffic = matrix;              (new TrafficPattern)
    injection_process = matrix;    (new InjectionProcess)

The file holds one ``src dst weight`` triple per line (``#`` comments).
Each source injects with probability ``injection_rate * w_src / W_total``
per cycle (so ``injection_rate`` is the total offered load of the whole
network) and picks its destination with probability ``w_sd / w_src``.

Usage:  python patch_booksim.py <booksim2 checkout>   (idempotent), then
        make -C <booksim2>/src
"""
import os
import re
import sys

MARK = "/* spicecimx-matrix */"

TRAFFIC_HPP = MARK + """
class MatrixTrafficPattern : public TrafficPattern {
protected:
  vector<vector<int> > _dst;
  vector<vector<double> > _cum;
public:
  MatrixTrafficPattern(int nodes, string const & file);
  virtual int dest(int source);
};
"""

TRAFFIC_CPP = MARK + """
#include <fstream>
#include <sstream>
MatrixTrafficPattern::MatrixTrafficPattern(int nodes, string const & file)
  : TrafficPattern(nodes), _dst(nodes), _cum(nodes)
{
  ifstream in(file.c_str());
  if(!in) { cout << "Error: cannot open matrix_file " << file << endl; exit(-1); }
  string line;
  while(getline(in, line)) {
    if(line.empty() || line[0] == '#') continue;
    istringstream ss(line);
    int s, d; double w;
    if(!(ss >> s >> d >> w) || w <= 0.0) continue;
    if(s < 0 || s >= nodes || d < 0 || d >= nodes) {
      cout << "Error: matrix_file node out of range: " << line << endl; exit(-1);
    }
    double prev = _cum[s].empty() ? 0.0 : _cum[s].back();
    _dst[s].push_back(d);
    _cum[s].push_back(prev + w);
  }
}

int MatrixTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes));
  if(_dst[source].empty()) return source;
  double r = RandomFloat() * _cum[source].back();
  for(size_t i = 0; i < _cum[source].size(); ++i)
    if(r < _cum[source][i]) return _dst[source][i];
  return _dst[source].back();
}
"""

INJ_HPP = MARK + """
class MatrixInjectionProcess : public InjectionProcess {
protected:
  vector<double> _node_rate;
public:
  MatrixInjectionProcess(int nodes, double rate, string const & file);
  virtual bool test(int source);
};
"""

INJ_CPP = MARK + """
#include <fstream>
#include <sstream>
MatrixInjectionProcess::MatrixInjectionProcess(int nodes, double rate,
                                               string const & file)
  : InjectionProcess(nodes, rate), _node_rate(nodes, 0.0)
{
  ifstream in(file.c_str());
  if(!in) { cout << "Error: cannot open matrix_file " << file << endl; exit(-1); }
  string line; double total = 0.0;
  while(getline(in, line)) {
    if(line.empty() || line[0] == '#') continue;
    istringstream ss(line);
    int s, d; double w;
    if(!(ss >> s >> d >> w) || w <= 0.0) continue;
    if(s >= 0 && s < nodes) { _node_rate[s] += w; total += w; }
  }
  for(int n = 0; n < nodes; ++n) {
    _node_rate[n] = (total > 0.0) ? rate * _node_rate[n] / total : 0.0;
    if(_node_rate[n] > 1.0) _node_rate[n] = 1.0;
  }
}

bool MatrixInjectionProcess::test(int source)
{
  assert((source >= 0) && (source < _nodes));
  return (_node_rate[source] > 0.0) && (RandomFloat() < _node_rate[source]);
}
"""


def _patch(path, fn):
    with open(path) as f:
        src = f.read()
    if MARK in src:
        return False
    new = fn(src)
    with open(path, "w") as f:
        f.write(new)
    return True


def _before_last_endif(text, block):
    head, sep, tail = text.rpartition("#endif")
    return head + block + "\n" + sep + tail


def main(root):
    s = os.path.join(root, "src")
    done = []
    done.append(_patch(os.path.join(s, "traffic.hpp"),
                       lambda t: _before_last_endif(t, TRAFFIC_HPP)))
    done.append(_patch(os.path.join(s, "traffic.cpp"), lambda t: re.sub(
        r'(\s*\}\s*else\s*\{\s*\n\s*cout << "Error: Unknown traffic pattern: ")',
        '\n  } else if(pattern_name == "matrix") {\n'
        '    result = new MatrixTrafficPattern(nodes, config->GetStr("matrix_file"));'
        r'\1', t, count=1) + "\n" + TRAFFIC_CPP))
    done.append(_patch(os.path.join(s, "injection.hpp"),
                       lambda t: _before_last_endif(t, INJ_HPP)))
    done.append(_patch(os.path.join(s, "injection.cpp"), lambda t: re.sub(
        r'(\s*\}\s*else\s*\{\s*\n\s*cout << "Invalid injection process: ")',
        '\n  } else if(process_name == "matrix") {\n'
        '    result = new MatrixInjectionProcess(nodes, load, config->GetStr("matrix_file"));'
        r'\1', t, count=1) + "\n" + INJ_CPP))
    done.append(_patch(os.path.join(s, "booksim_config.cpp"), lambda t: t.replace(
        'AddStrField("tech_file", "");',
        'AddStrField("tech_file", "");\n  AddStrField("matrix_file", ""); ' + MARK, 1)))
    print("patched" if any(done) else "already patched", root)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "booksim2")
