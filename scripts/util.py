#!/usr/bin/env python3
"""Мелкий общий помощник для навыка: файл читается и закрывается сразу.

Прежде `json.load(open(path))` и `open(path).read()` оставляли дескриптор открытым:
при частых вызовах (маркеры, классификатор, MCP-сервер) это накапливает утечку
файловых дескрипторов на машине, где лимит 256 и утечка уже была замечена.
"""
import json

def load_json(path, encoding="utf-8"):
    with open(path, encoding=encoding) as f:
        return json.load(f)

def read_text(path, encoding="utf-8", errors="replace"):
    with open(path, encoding=encoding, errors=errors) as f:
        return f.read()

def read_lines(path, encoding="utf-8"):
    with open(path, encoding=encoding) as f:
        return [l for l in f if l.strip()]

def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()

# Рабочая точка правила: ⚫ ставится при p ≥ порога метки И p ≥ 0.70. Одна для всех
# путей (CLI, worker, плагин, psych_layer) — раньше каждый держал свою, и CLI без --min
# применял только thresholds.npy (14 из 23 порогов ниже 0.70), то есть размечал слабее,
# чем плагин: тот же текст давал разные вердикты в зависимости от пути.
OPERATING_POINT = 0.70

def write_json(path, obj, encoding="utf-8", indent=1):
    with open(path, "w", encoding=encoding) as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
