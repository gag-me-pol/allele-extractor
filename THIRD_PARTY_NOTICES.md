# Third-Party Notices

This project bundles and/or depends on third-party open-source software. This file lists each component and its license so you (and anyone you share this repository with) know what you're redistributing.

This is informational, not legal advice. If you have any doubt about a specific use case (especially commercial or closed-source use), check the linked license texts and, if needed, consult a lawyer.

## Bundled binaries (shipped inside this repository)

### Tesseract OCR — `Tesseract-OCR/`

- Project: https://github.com/tesseract-ocr/tesseract
- License: **Apache License 2.0**
- Full license text bundled at: `Tesseract-OCR/doc/LICENSE`
- Authors: `Tesseract-OCR/doc/AUTHORS`
- Includes the English trained data file `Tesseract-OCR/tessdata/eng.traineddata`, also Apache License 2.0.
- The build also includes a few Java `.jar` files (`ScrollView.jar`, `piccolo2d-*.jar`, `jaxb-api-2.3.1.jar`) that are part of Tesseract's official Windows distribution (used only by its training/debugging GUI, not by this project's code). They carry their own upstream licenses (BSD/Apache family) — not used or invoked by this project.

Apache 2.0 is a permissive license: it allows use, modification, and redistribution (including commercially), and only requires that you keep copyright/license notices and state any changes you made to the licensed code. It does **not** require your own code to be open-sourced.

### Python 3.14.6 (portable/embeddable build) — `python-3.14.6/`

- Project: https://www.python.org/
- License: **PSF License Agreement** (Python Software Foundation License), a permissive BSD-style license
- Full license text bundled at: `python-3.14.6/LICENSE.txt`

## Python packages (installed under `python-3.14.6/Lib/site-packages`)

| Package | Version bundled | License | Notes |
|---|---|---|---|
| opencv-python | 4.13.0.92 | Apache License 2.0 | |
| **PyMuPDF** (`fitz`) | 1.27.2.3 | **GNU AGPL v3.0**, or a commercial license from Artifex | ⚠️ see "PyMuPDF and AGPL" below |
| Pillow | 12.2.0 | HPND License (permissive) | |
| numpy | 2.4.6 | BSD 3-Clause | |
| pandas | 3.0.3 | BSD 3-Clause | |
| openpyxl | 3.1.5 | MIT License | |
| pypdf | 6.13.2 | BSD 3-Clause | |
| pytesseract | 0.3.13 | Apache License 2.0 | |
| thefuzz | 0.22.1 | MIT License | |
| rapidfuzz | 3.14.5 | MIT License | matching backend used by thefuzz |
| python-dateutil | 2.9.0.post0 | Apache 2.0 / BSD (dual-licensed) | pandas dependency |
| six | 1.17.0 | MIT License | dateutil dependency |
| et_xmlfile | 2.0.0 | MIT License | openpyxl dependency |
| packaging | 26.2 | Apache 2.0 / BSD (dual-licensed) | |
| tzdata | 2026.3 | Apache License 2.0 | |

All of the above except PyMuPDF are permissive licenses (MIT, BSD, Apache 2.0, HPND, PSF). They allow commercial and non-commercial use, modification, and redistribution, and only require keeping the original copyright/license notices somewhere in the distribution — they do not require this project's own code to be open-sourced or licensed the same way.

## PyMuPDF and AGPL (resolved: project is now AGPL-3.0)

This project uses **PyMuPDF** (`fitz`) to render PDF pages to images. PyMuPDF is dual-licensed by its maintainer, Artifex Software:

- **GNU Affero General Public License v3.0 (AGPL-3.0)** — free to use, but a strong copyleft license, or
- a **commercial license purchased from Artifex** — for use in proprietary/closed-source software without AGPL obligations.

Because this project uses the free (AGPL) side of that dual license rather than a paid commercial license, the project's own code is now licensed under **AGPL-3.0-or-later** too (see the top-level [`LICENSE`](LICENSE) file) — this keeps the whole combined work under one consistent copyleft license, which is required precisely because AGPL-licensed code is included.

Practical implication: this project — and anything built from it — must stay AGPL-licensed (or be re-evaluated) for as long as it depends on PyMuPDF's AGPL license. If a future version needs a permissive license again, the options remain the same as before: buy a commercial PyMuPDF license from Artifex, or replace PyMuPDF with a permissively-licensed alternative such as `pypdfium2` (BSD/Apache) or `pdf2image` (MIT, backed by Poppler — check Poppler's own license before choosing this route, since Poppler itself is GPL-licensed).

## A note on repository size

The bundled Python and Tesseract folders contain large compiled binaries (Tesseract's OCR engine and DLLs, its English language model, and compiled Python packages such as OpenCV, NumPy and pandas). These are tracked with [Git LFS](https://git-lfs.com/) (see `.gitattributes` at the repository root) rather than committed directly, to keep the regular git history manageable. Anyone cloning the repository needs Git LFS installed (`git lfs install`) to pull the actual file contents rather than pointer stubs.
