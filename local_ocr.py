"""Small, bounded local OCR wrapper for low-memory deployments."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

_OCR_REQUEST = re.compile(
    r"\b(?:ocr|extract(?:\s+the)?\s+text|transcrib(?:e|ing)|document\s+parsing|read\s+(?:the\s+)?(?:text|scan))\b",
    re.IGNORECASE,
)


def _int_setting(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, default)))
    except ValueError:
        return default


def is_requested(text: str) -> bool:
    return bool(_OCR_REQUEST.search(text or ""))


def is_available() -> bool:
    return bool(shutil.which(os.getenv("LOCAL_OCR_COMMAND", "tesseract")))


def extract_image(data: bytes, suffix: str = ".png") -> str:
    command = os.getenv("LOCAL_OCR_COMMAND", "tesseract")
    if not shutil.which(command):
        raise RuntimeError("Tesseract is not installed")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as source:
        source.write(data)
        source.flush()
        result = subprocess.run(
            [command, source.name, "stdout", "-l", os.getenv("LOCAL_OCR_LANGUAGE", "eng")],
            capture_output=True,
            text=True,
            timeout=_int_setting("LOCAL_OCR_TIMEOUT_SECONDS", 30),
            check=False,
        )
    if result.returncode:
        raise RuntimeError(result.stderr.strip()[:200] or "Tesseract failed")
    return result.stdout.strip()


def extract_pdf(data: bytes) -> str:
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError("Local PDF OCR requires PyMuPDF") from error
    document = fitz.open(stream=data, filetype="pdf")
    pages = []
    try:
        dpi = _int_setting("LOCAL_OCR_PDF_DPI", 150)
        for page_index in range(min(document.page_count, _int_setting("LOCAL_OCR_MAX_PAGES", 5))):
            pixmap = document[page_index].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
            text = extract_image(pixmap.tobytes("png"))
            if text:
                pages.append(f"--- Page {page_index + 1} ---\n{text}")
    finally:
        document.close()
    return "\n\n".join(pages)
