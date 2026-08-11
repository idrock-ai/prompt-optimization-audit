#!/bin/bash
set -euo pipefail
cd ~/prompt-optimization-audit
python3 --version            # expect 3.12.x
python3 -m venv .venv
.venv/bin/pip install -q -r requirements-dev.txt
.venv/bin/python -m pytest tests/ -q
