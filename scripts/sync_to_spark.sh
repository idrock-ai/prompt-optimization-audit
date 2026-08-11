#!/bin/bash
set -euo pipefail
rsync -av --delete --exclude .git --exclude .venv --exclude __pycache__ \
  --exclude results ./ newuu_3@spark-3.idrock.uz:~/prompt-optimization-audit/
