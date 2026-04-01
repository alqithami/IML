#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# ONE-SHOT FIX: Completely resets and patches SSD to work without Ray.
# Run from the IML-main directory:   bash fix_ssd.sh
# ============================================================================

echo "=== Step 1: Removing corrupted SSD clone ==="
rm -rf sequential_social_dilemma_games

echo "=== Step 2: Cloning fresh copy ==="
git clone https://github.com/eugenevinitsky/sequential_social_dilemma_games.git

cd sequential_social_dilemma_games

echo "=== Step 3: Copying utility_funcs.py into the package ==="
cp utility_funcs.py social_dilemmas/utility_funcs.py

echo "=== Step 4: Patching agent.py (utility_funcs import) ==="
sed -i.bak 's/^import utility_funcs as util$/from social_dilemmas import utility_funcs as util/' social_dilemmas/envs/agent.py

echo "=== Step 5: Patching map_env.py (Ray/RLlib imports) ==="
python3 << 'PATCH1'
with open("social_dilemmas/envs/map_env.py", "r") as f:
    content = f.read()

content = content.replace(
    "from ray.rllib.agents.callbacks import DefaultCallbacks\nfrom ray.rllib.env import MultiAgentEnv",
    """# --- Ray/RLlib stubs (IML patch: works without Ray) ---
try:
    from ray.rllib.agents.callbacks import DefaultCallbacks
except (ImportError, ModuleNotFoundError):
    class DefaultCallbacks:
        pass

try:
    from ray.rllib.env import MultiAgentEnv
except (ImportError, ModuleNotFoundError):
    import gym
    class MultiAgentEnv(gym.Env):
        pass
# --- End stubs ---"""
)

with open("social_dilemmas/envs/map_env.py", "w") as f:
    f.write(content)
print("  map_env.py patched OK")
PATCH1

echo "=== Step 6: Patching switch.py (Ray/RLlib import) ==="
python3 << 'PATCH2'
with open("social_dilemmas/envs/switch.py", "r") as f:
    content = f.read()

content = content.replace(
    "from ray.rllib.agents.callbacks import DefaultCallbacks",
    """# --- Ray/RLlib stub (IML patch) ---
try:
    from ray.rllib.agents.callbacks import DefaultCallbacks
except (ImportError, ModuleNotFoundError):
    class DefaultCallbacks:
        pass
# --- End stub ---"""
)

with open("social_dilemmas/envs/switch.py", "w") as f:
    f.write(content)
print("  switch.py patched OK")
PATCH2

echo "=== Step 7: Cleaning up backup files ==="
find social_dilemmas/ -name "*.bak" -delete 2>/dev/null || true

echo "=== Step 8: Installing (editable, no deps) ==="
pip install --no-deps -e .

cd ..

echo ""
echo "=== Step 9: Verifying ==="
python3 -c "
from social_dilemmas.envs.cleanup import CleanupEnv
env = CleanupEnv(num_agents=5)
obs = env.reset()
print(f'  CleanupEnv: {len(obs)} agents')

from social_dilemmas.envs.harvest import HarvestEnv
env2 = HarvestEnv(num_agents=5)
obs2 = env2.reset()
print(f'  HarvestEnv: {len(obs2)} agents')

print()
print('=== SSD installation successful! ===')
"
