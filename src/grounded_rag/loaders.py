"""Turn files into ``Document`` objects: Markdown (with YAML frontmatter), text, DOCX, PDF.

DOCX and PDF support are optional extras so the core install stays small.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

import yaml

from grounded_rag.envelopes import Document

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".docx", ".pdf"}

_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_MD_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)


class UnsupportedFormat(ValueError):
    """The file type is not supported, or its optional extra is not installed."""


def slugify(name: str) -> str:
    """A document id: lowercase, [a-z0-9._-], at most 64 characters."""
    slug = re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-._")
    return slug[:64] or "doc"


def _optional(module: str, extra: str) -> ModuleType:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise UnsupportedFormat(
            f"{module} is not installed: pip install 'grounded-rag-pipeline[{extra}]'"
        ) from exc


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text
    data = yaml.safe_load(match.group(1)) or {}
    if not isinstance(data, Mapping):
        return {}, text
    return {str(k): str(v) for k, v in data.items()}, text[match.end() :]


def _read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8")
    if suffix == ".docx":
        docx = _optional("docx", "docx")
        paragraphs = docx.Document(str(path)).paragraphs
        return "\n\n".join(p.text for p in paragraphs if p.text.strip())
    if suffix == ".pdf":
        pypdf = _optional("pypdf", "pdf")
        pages = pypdf.PdfReader(str(path)).pages
        return "\n\n".join(text for page in pages if (text := page.extract_text() or "").strip())
    raise UnsupportedFormat(f"unsupported file type {suffix.lstrip('.') or '(none)'}: {path.name}")


def load_path(path: Path, *, metadata: Mapping[str, str] | None = None) -> Document:
    """Load one file. Frontmatter keys become metadata; the first heading becomes the title."""
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise UnsupportedFormat(f"unsupported file type {path.suffix.lstrip('.')}: {path.name}")
    front, text = _split_frontmatter(_read_text(path))
    heading = _MD_HEADING.search(text)
    stem_title = re.sub(r"[_-]+", " ", path.stem).strip().capitalize()
    return Document(
        doc_id=slugify(path.stem),
        title=heading.group(1) if heading else stem_title,
        text=text.strip(),
        metadata={**front, **(metadata or {})},
    )


def load_directory(directory: Path) -> tuple[list[Document], list[Path]]:
    """Load every supported file (sorted by name); return the documents and the skipped paths."""
    documents, skipped = [], []
    for path in sorted(Path(directory).iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            documents.append(load_path(path))
        else:
            skipped.append(path)
    return documents, skipped
