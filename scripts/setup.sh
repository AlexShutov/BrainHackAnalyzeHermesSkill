#!/usr/bin/env bash
# Install the psychology-analysis skill on a clean Hermes.
# Run after cloning the repo into ~/.hermes/skills/message-text-analysis.
set -euo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
SKILL="$HERMES_HOME/skills/message-text-analysis"
VENV="$HERMES_HOME/tools/psych-rag-env"

echo "1/4 venv: $VENV"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet numpy onnxruntime tokenizers huggingface_hub requests
# torch is an export-time dependency only — the runtime path never imports it
"$VENV/bin/pip" install --quiet torch transformers

echo "2/4 weights: download + convert to ONNX int8"
"$VENV/bin/pip" show torch >/dev/null 2>&1 || echo "torch needed only for conversion; runtime uses onnxruntime"
"$VENV/bin/python" "$SKILL/scripts/model_setup.py"

echo "3/4 RAG index"
"$VENV/bin/python" "$SKILL/scripts/rag_build.py"

echo "4/4 plugin"
mkdir -p "$HERMES_HOME/plugins/psych-rag"
cp "$SKILL/plugin/psych-rag/__init__.py" "$SKILL/plugin/psych-rag/plugin.yaml" "$HERMES_HOME/plugins/psych-rag/"

cat <<'EOF'

Добавьте MCP-сервер в config.yaml:

  mcp_servers:
    psychology-rag:
      command: <HERMES_HOME>/tools/psych-rag-env/bin/python
      args: ["<HERMES_HOME>/skills/message-text-analysis/scripts/rag_mcp.py"]
      enabled: true

и перезапустите Hermes (плагины и MCP грузятся на старте).

Книга в репозиторий не включена (авторские права): без неё индекс строится только по
devices.json — навык работает, но retrieval даёт только определения приёмов.
EOF
