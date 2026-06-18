"""Limpieza de transcripciones: HTML/timestamps → texto plano.

El scraper extrae los párrafos de HappyScribe como una lista de bloques con
timestamp. Este módulo los normaliza a texto plano legible y cuenta palabras,
sin lógica de red ni de base de datos (puro, fácil de testear).
"""

from __future__ import annotations

import re

# [HH:MM:SS], [HH:MM:SS.ss] o un timestamp suelto al inicio de línea "HH:MM:SS"
_TIMESTAMP_RE = re.compile(r"\[?\b\d{1,2}:\d{2}:\d{2}(?:\.\d{1,3})?\]?")
_MULTISPACE_RE = re.compile(r"[ \t]+")
_MULTINEWLINE_RE = re.compile(r"\n{3,}")


def strip_timestamps(text: str) -> str:
    """Elimina marcas de tiempo tipo ``[00:01:23]`` / ``00:01:23`` del texto."""
    return _TIMESTAMP_RE.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    """Colapsa espacios redundantes, preserva saltos de párrafo."""
    text = _MULTISPACE_RE.sub(" ", text)
    # limpia espacios al borde de cada línea
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _MULTINEWLINE_RE.sub("\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Pipeline completo: quita timestamps y normaliza espacios."""
    return normalize_whitespace(strip_timestamps(text))


def paragraphs_to_text(paragraphs: list[dict]) -> str:
    """Convierte la lista de párrafos del scraper en texto plano.

    Cada párrafo es ``{"seconds": int, "timestamp": str, "text": str}``.
    Une los textos con saltos de línea y los limpia.
    """
    lines = [clean_text(p.get("text", "")) for p in paragraphs]
    body = "\n".join(line for line in lines if line)
    return normalize_whitespace(body)


def count_words(text: str) -> int:
    """Conteo simple de palabras sobre texto plano."""
    return len(text.split())
