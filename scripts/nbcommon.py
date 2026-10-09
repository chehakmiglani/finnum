"""Shared helpers for building self-contained Colab notebooks."""
import base64
import io
import zipfile
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

SETUP = """import os, sys, io, base64, zipfile, torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "NONE - Runtime > Change runtime type > T4 GPU")
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    RESULTS = "/content/drive/MyDrive/finnum_results"
else:
    RESULTS = "/content/finnum_results"
os.makedirs(RESULTS, exist_ok=True)

# The finnum package is embedded below, so nothing needs uploading
FINNUM_ZIP = "__FINNUM_ZIP_B64__"
zipfile.ZipFile(io.BytesIO(base64.b64decode(FINNUM_ZIP))).extractall("/content/finnum")
sys.path.insert(0, "/content/finnum/src")
!pip -q install __PIP__
import finnum
print("finnum", finnum.__version__, "| results ->", RESULTS)"""


def setup_cell(pip: str):
    return code(SETUP.replace("__PIP__", pip))


def package_b64() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted((ROOT / "src").rglob("*.py")):
            zf.write(f, f.relative_to(ROOT))
        zf.write(ROOT / "pyproject.toml", "pyproject.toml")
    return base64.b64encode(buf.getvalue()).decode()


def write(cells, name: str) -> Path:
    b64 = package_b64()
    for c in cells:
        if c.cell_type == "code" and "__FINNUM_ZIP_B64__" in c.source:
            c.source = c.source.replace("__FINNUM_ZIP_B64__", b64)
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}})
    out = ROOT / "notebooks" / name
    nbf.write(nb, out)
    print("wrote", out)
    return out
