#!/usr/bin/env bash
# Create a virtual environment for kmeans_in_production.ipynb and kmeans_production.py, wherever the repo is cloned.
#
#   ./setup.sh                 create ./.venv next to this script and install requirements.txt
#   ./setup.sh --kernel        also register the venv as a Jupyter kernel, "k-means production (.venv)"
#   ./setup.sh --run           also execute the notebook headlessly (about a minute) to prove the setup works
#   PYTHON=/path/to/python3.12 ./setup.sh     use that interpreter (3.10 or newer; the default is python3)
#   VENV_DIR=/elsewhere/.venv ./setup.sh      put the venv somewhere else
#
# Re-running is safe: an existing venv is reused and only missing packages are installed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-$HERE/.venv}"
KERNEL=0
RUN=0
for arg in "$@"; do
  case "$arg" in
    --kernel) KERNEL=1 ;;
    --run) RUN=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

# 1. the interpreter
command -v "$PYTHON" >/dev/null 2>&1 || { echo "cannot find $PYTHON; set PYTHON=/path/to/python3" >&2; exit 1; }
"$PYTHON" - <<'EOF'
import sys
if sys.version_info < (3, 10):
    sys.exit(f"Python 3.10 or newer is needed, found {sys.version.split()[0]}: run with PYTHON=/path/to/python3.12 ./setup.sh")
print(f"using {sys.executable} ({sys.version.split()[0]})")
EOF

# 2. the venv
if [ -x "$VENV_DIR/bin/python" ]; then
  echo "reusing $VENV_DIR"
else
  "$PYTHON" -m venv "$VENV_DIR"
  echo "created $VENV_DIR"
fi
PY="$VENV_DIR/bin/python"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -r "$HERE/requirements.txt"

# 3. smoke test: the libraries import and the module loads from this folder
cd "$HERE"
"$PY" - <<'EOF'
import matplotlib, numpy, scipy, sklearn
import kmeans_production
print(f"numpy {numpy.__version__}, scipy {scipy.__version__}, scikit-learn {sklearn.__version__}, "
      f"matplotlib {matplotlib.__version__}; kmeans_production imports")
EOF

# 4. a kernel visible to Jupyter and VS Code (optional; VS Code can also pick the venv's python directly)
if [ "$KERNEL" = 1 ]; then
  "$PY" -m ipykernel install --user --name kmeans-production --display-name "k-means production (.venv)"
fi

# 5. execute the notebook headlessly (optional). The kernel is registered inside the venv so no other Python can
#    shadow it; the notebook is written back with its outputs.
if [ "$RUN" = 1 ]; then
  "$PY" -m ipykernel install --sys-prefix --name kmeans-production >/dev/null
  "$PY" - <<'EOF'
import time
import nbformat
from nbclient import NotebookClient
nb = nbformat.read("kmeans_in_production.ipynb", as_version=4)
t = time.perf_counter()
NotebookClient(nb, timeout=1200, kernel_name="kmeans-production", resources={"metadata": {"path": "."}}).execute()
nbformat.write(nb, "kmeans_in_production.ipynb")
print(f"kmeans_in_production.ipynb executed in {time.perf_counter() - t:.0f} s; outputs saved")
EOF
fi

echo
echo "next: open kmeans_in_production.ipynb in VS Code and pick the interpreter $PY as its kernel,"
echo "      or: $PY -m pip install jupyterlab && $PY -m jupyterlab"
