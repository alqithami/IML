#!/usr/bin/env bash
set -euo pipefail

# Convenience installer for the SSD environments (Harvest/Cleanup) WITHOUT Ray/RLlib.
#
# Why this exists:
# - The SSD reference repo pins ray[rllib]==0.8.5 in requirements.txt.
# - Ray 0.8.5 is generally not installable on modern systems.
# - This pipeline does NOT use RLlib, so we exclude Ray.
#
# IMPORTANT:
# - The SSD package metadata enforces python<3.10.

# This script also performs a small cleanup step because many users
# accidentally install `social-dilemmas` from PyPI (or install the SSD repo
# *with* dependencies), which triggers an attempt to install `ray==0.8.5`.
# Ray 0.8.5 is not available for most modern Python/macOS builds.

REPO_URL="https://github.com/eugenevinitsky/sequential_social_dilemma_games"
TARGET_DIR="${1:-sequential_social_dilemma_games}"

# --- Python version guard ---
python - <<'PY'
import sys
maj, min = sys.version_info[:2]
if (maj, min) >= (3, 10):
    raise SystemExit(
        f"ERROR: SSD requires Python < 3.10, but you are on {sys.version.split()[0]}\n"
        "Create a Python 3.9 environment (e.g., conda create -n imlssd python=3.9) and rerun."
    )
print(f"[install_ssd_no_ray] Python OK: {sys.version.split()[0]}")
PY

# --- Cleanup potentially broken installs (safe to ignore if not present) ---
echo "[install_ssd_no_ray] Cleaning any previous 'social-dilemmas' / 'ray' installs (safe if not installed)..."
python -m pip uninstall -y social-dilemmas social_dilemmas sequential-social-dilemma-games ray rllib 2>/dev/null || true
python -m pip uninstall -y dm_tree 2>/dev/null || true
python -m pip install --upgrade pip setuptools wheel

if [[ -d "$TARGET_DIR" ]]; then
  echo "[install_ssd_no_ray] Target directory already exists: $TARGET_DIR"
else
  git clone "$REPO_URL" "$TARGET_DIR"
fi

cd "$TARGET_DIR"

# Install all SSD deps except Ray/RLlib.
if [[ -f requirements.txt ]]; then
  grep -v '^ray' requirements.txt > requirements_no_ray.txt
  python -m pip install -r requirements_no_ray.txt
else
  echo "[install_ssd_no_ray] WARNING: requirements.txt not found; installing a minimal dependency set."
  python -m pip install gym>=0.21.0 pettingzoo>=1.13.1 opencv-python numpy scipy pandas matplotlib lz4 setproctitle boto3 psutil
fi

# Editable install without deps to avoid any pinned Ray requirement.
python -m pip install -e . --no-deps

echo "[install_ssd_no_ray] Done. Quick check:" \
  && python -c "from social_dilemmas.envs.cleanup import CleanupEnv; env=CleanupEnv(num_agents=5); env.reset(); print('CleanupEnv import OK')"
