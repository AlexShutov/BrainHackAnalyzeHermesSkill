#!/usr/bin/env bash
# Install the psychology-analysis skill on a clean Hermes.
# Run after cloning the repo (any folder name) into $HERMES_HOME/skills/.
set -euo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
SKILL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PSYCH_RAG_VENV:-$HERMES_HOME/tools/psych-rag-env}"

echo "1/4 venv: $VENV"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install --quiet numpy onnxruntime tokenizers huggingface_hub requests
# torch is an export-time dependency only — the runtime path never imports it.
# Skip it with SKIP_TORCH=1 if it is already installed or the model ships an ONNX export.
if [ "${SKIP_TORCH:-0}" != "1" ]; then
  "$VENV/bin/pip" install --quiet torch transformers
else
  echo "  torch пропущен (SKIP_TORCH=1) — экспорт не выполнится, нужен готовый ONNX-экспорт модели"
fi

echo "2/4 weights: download + convert to ONNX int8"
"$VENV/bin/python" "$SKILL/scripts/model_setup.py"

echo "3/4 RAG index"
"$VENV/bin/python" "$SKILL/scripts/rag_build.py"

echo "4/4 plugin"
mkdir -p "$HERMES_HOME/plugins/psych-rag"
cp "$SKILL/plugin/psych-rag/__init__.py" "$SKILL/plugin/psych-rag/plugin.yaml" "$HERMES_HOME/plugins/psych-rag/"

cat <<EOF

Добавьте MCP-сервер в config.yaml:

  mcp_servers:
    psychology-rag:
      command: $VENV/bin/python
      args: ["$SKILL/scripts/rag_mcp.py"]
      enabled: true

и перезапустите Hermes (плагины и MCP грузятся на старте).

Книга в репозиторий не включена (авторские права): без неё индекс строится только по
devices.json — навык работает, но retrieval даёт только определения приёмов.
EOF
