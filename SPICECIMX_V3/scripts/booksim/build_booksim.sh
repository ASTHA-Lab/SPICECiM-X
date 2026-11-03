#!/bin/bash
# Clone BookSim2, add the SPICECiM-X trace-matrix traffic mode, build it.
#   bash build_booksim.sh [target_dir]      (default ./booksim2)
# Then set [INTERCONNECT] booksim_binary = <target_dir>/src/booksim
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
DST="${1:-booksim2}"
if [ ! -d "$DST" ]; then
  git clone https://github.com/booksim/booksim2.git "$DST"
fi
# Pin the commit the patch was written against (upstream master, 2017-06-27).
git -C "$DST" checkout -q 28f43299f1706a3160ffac721ca461d74eb6e618
python3 "$HERE/patch_booksim.py" "$DST"
make -C "$DST/src" -j"$(nproc 2>/dev/null || echo 4)"
echo "built: $(cd "$DST/src" && pwd)/booksim"
