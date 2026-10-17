#!/usr/bin/env bash
# Install the psychology-analysis skill on a clean Hermes.
# Run after cloning the repo (any folder name) into $HERMES_HOME/skills/.
set -euo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
SKILL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PSYCH_RAG_VENV:-$HERMES_HOME/tools/psych-rag-env}"

# onnxruntime ships wheels only for CPython 3.11-3.14; a newer interpreter breaks the install.
pick_python() {
  for p in python3.13 python3.12 python3.11 python3.14 python3; do
    command -v "$p" >/dev/null || continue
    v="$($p -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    case "$v" in 3.11|3.12|3.13|3.14) echo "$p"; return 0;; esac
  done
  return 1
}

PYBIN="$(pick_python || true)"
if [ -z "$PYBIN" ]; then
  echo "Нужен CPython 3.11-3.14 (onnxruntime не имеет колёс для более новых версий)." >&2
  echo "Текущий python3: $(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo '?')" >&2
  exit 1
fi

echo "1/4 venv: $VENV (python $PYBIN)"
if [ ! -x "$VENV/bin/python" ]; then
  "$PYBIN" -m venv "$VENV"
fi
"$VENV/bin/pip" install --quiet numpy onnxruntime tokenizers huggingface_hub requests
# torch + onnx are export-time dependencies only — the runtime path never imports them.
# Skip them with SKIP_TORCH=1 if already installed or if the model ships an ONNX export.
if [ "${SKIP_TORCH:-0}" != "1" ]; then
  "$VENV/bin/pip" install --quiet torch transformers onnx
else
  echo "  torch/onnx пропущены (SKIP_TORCH=1) — экспорт не выполнится, нужен готовый ONNX-экспорт модели"
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
