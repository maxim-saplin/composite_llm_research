#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="."
ENV_FILE="$ROOT_DIR/.env"

if [[ -f "$ENV_FILE" ]]; then
  # Export all variables from .env (ignores comments/blank lines)
  set -a
  source "$ENV_FILE"
  set +a
else
  echo "ERROR: .env not found at $ENV_FILE" >&2
  exit 1
fi

# Convenience: if only LITELLM_API_KEY is set, reuse it for Cerebras
if [[ -z "${CEREBRAS_API_KEY:-}" && -n "${LITELLM_API_KEY:-}" ]]; then
  export CEREBRAS_API_KEY="$LITELLM_API_KEY"
fi

exec litellm --config "$ROOT_DIR/litellm_proxy.example.yaml" --host 0.0.0.0 --port 4000
