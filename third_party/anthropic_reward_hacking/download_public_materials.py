"""Download public materials for Anthropic's reward-hacking EM paper.

This script intentionally downloads only public web artifacts:
- official Anthropic PDF
- Anthropic research page
- arXiv abstract page

No official code or machine-readable benchmark dataset was found at the time
this directory was created.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent

DOWNLOADS = {
    ROOT / "paper" / "natural_emergent_misalignment_reward_hacking.pdf": (
        "https://assets.anthropic.com/m/74342f2c96095771/original/"
        "Natural-emergent-misalignment-from-reward-hacking-paper.pdf"
    ),
    ROOT / "sources" / "anthropic_research_page.html": (
        "https://www.anthropic.com/research/emergent-misalignment-reward-hacking"
    ),
    ROOT / "sources" / "arxiv_abs_2511_18397.html": "https://arxiv.org/abs/2511.18397",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())
    print(f"{path}: {sha256(path)}")


def extract_pdf_text(pdf_path: Path, txt_path: Path) -> None:
    try:
        from pypdf import PdfReader
    except ImportError:
        print("Skipping text extraction: install pypdf first.")
        return

    reader = PdfReader(str(pdf_path))
    pages = []
    for page_idx, page in enumerate(reader.pages, start=1):
        pages.append(f"\n\n===== PAGE {page_idx} =====\n{page.extract_text() or ''}")
    txt_path.write_text("\n".join(pages), encoding="utf-8")
    print(f"{txt_path}: extracted {len(reader.pages)} pages")


def main() -> None:
    for path, url in DOWNLOADS.items():
        download(url, path)

    extract_pdf_text(
        ROOT / "paper" / "natural_emergent_misalignment_reward_hacking.pdf",
        ROOT / "paper" / "natural_emergent_misalignment_reward_hacking.txt",
    )


if __name__ == "__main__":
    main()
