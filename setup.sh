#!/usr/bin/env bash
set -Eeuo pipefail

# Minimal setup for Vast.ai:
# - Install uv (if missing)
# - Put uv on PATH now and persistently
# - Install Python 3.10 with uv
# - Create venv (if missing), activate it
# - uv sync (uses pyproject.toml + uv.lock)

REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
cd "$REPO_DIR"

# 0) Sanity: require pyproject.toml
if [[ ! -f "pyproject.toml" ]]; then
  echo "ERROR: pyproject.toml not found in $REPO_DIR" >&2
  exit 1
fi

# 1) Install uv if not present
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# 2) Make sure uv is available in THIS shell
if [[ -f "$HOME/.local/bin/env" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/.local/bin/env"
else
  export PATH="$HOME/.local/bin:$PATH"
fi

# 3) Persist PATH for future shells (idempotent append)
persist_env_helper() {
  local rc="$1"
  [[ -f "$rc" ]] || return 0
  if ! grep -q 'source "\?$HOME/.local/bin/env"\?' "$rc" 2>/dev/null; then
    printf '\n# Ensure uv is on PATH\n[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"\n' >> "$rc"
  fi
}
persist_env_helper "$HOME/.bashrc"
persist_env_helper "$HOME/.zshrc"
persist_env_helper "$HOME/.profile"
persist_env_helper "$HOME/.bash_profile"

# --- Install system dependencies for OpenCV and matplotlib ---
echo "🧩 Installing system dependencies..."
apt-get update -y
apt-get install -y libgl1 libglib2.0-0

# 4) Ensure Python 3.10 via uv and create venv if missing
uv python install 3.10

if [[ -d ".venv" ]]; then
  echo "Using existing .venv"
else
  echo "Creating .venv with Python 3.10..."
  uv venv --python 3.10
fi

# 5) Activate venv
# shellcheck source=/dev/null
source .venv/bin/activate

# --- Add uv back to PATH inside venv ---
export PATH="$HOME/.local/bin:$PATH"

# --- Sync dependencies ---echo "Syncing dependencies with uv..."
uv sync

echo
echo "✅ Setup complete."
echo "To activate later:  source \"$REPO_DIR/.venv/bin/activate\""