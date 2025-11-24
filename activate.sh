# Add project root to PYTHONPATH so Python can find the agentic module
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source .venv/bin/activate
