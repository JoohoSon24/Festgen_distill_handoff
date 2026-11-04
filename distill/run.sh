#!/usr/bin/env bash
# Resumable launcher: rerunning continues from the last saved stage/update.
cd "$(dirname "$0")/.."
/root/Diffusion/workspace_distillation/.venv/bin/python -B -u distill/pd_dsm.py --run main "$@" >> distill/runs/main.log 2>&1
echo "exit $? $(date -u)" >> distill/runs/main.log
