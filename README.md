# OCR Document Organiser

Hundreds of scanned documents — insurance letters, financial statements, bank correspondence — sitting in a folder as `scan_001.pdf`, `scan_002.pdf`. Finding anything is a manual slog. This tool eliminates that problem.

Drop PDFs into a watched folder. The pipeline runs OCR locally, sends only the extracted text to Google Gemini, and renames each file to a structured, human-readable name like `2024-03-15 - Aviva - Annuity Statement - A Smith - POL-00123456.pdf`. The result lands in an organised output folder, the original is archived, and a CSV audit trail is written automatically.

---

## How It Works

A three-stage pipeline runs per file:

1. **OCR (local)** — `pdf2image` renders each page; Tesseract extracts text. Nothing leaves your machine at this stage.
2. **AI naming (Gemini)** — Extracted text is sent to `gemini-2.0-flash` with a structured prompt. The model returns a filename in the format `YYYY-MM-DD - Company - Document_Type - Subject - Reference.pdf`. Raw document images are never uploaded.
3. **File organisation** — The renamed copy is written to `organised_files/`. The original is moved to `processing_completed/` or `processing_failed/` depending on outcome.

Progress streams back to the browser in real time via **Server-Sent Events** — no polling, no page refreshes.

---

## Features

- **Privacy-conscious architecture** — OCR runs entirely on-device via Tesseract. Only extracted text (no images, no raw PDFs) is sent to the Gemini API.
- **Structured AI-generated filenames** — Date, company, document type, subject, and reference number extracted and formatted consistently.
- **Duplicate-safe output** — If a suggested filename already exists in `organised_files/`, a timestamp suffix is appended rather than overwriting.
- **Real-time progress streaming** — Flask SSE stream pushes per-file status to the browser as processing happens.
- **Automated archiving** — Processed originals are moved to `processing_completed/`; failures go to `processing_failed/` for review.
- **CSV audit log** — Each run writes a timestamped report to `reports/` with original name, extracted text, and new name.
- **Dockerised** — Single container bundles Python, Tesseract, and Poppler. No system dependency wrangling.

---

## Tech Stack

| Component | Technology |
|---|---|
| Runtime | Python 3.12 |
| Web framework | Flask 3.0 |
| OCR engine | Tesseract OCR (via `pytesseract`) |
| PDF rendering | `pdf2image` + Poppler |
| AI model | Google Gemini API (`gemini-2.0-flash`) |
| Data / reporting | pandas |
| Containerisation | Docker |

---

## Quick Start (Docker)

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

```bash
# 1. Clone the repo
git clone https://github.com/your-username/ocr-document-organiser.git
cd ocr-document-organiser

# 2. Set your Gemini API key
echo 'GEMINI_API_KEY="your_key_here"' > .env

# 3. Build and run
docker build -t ocr-document-organiser .

docker run -d \
  --env-file .env \
  -p 5000:5000 \
  -v ./input_files:/app/input_files \
  -v ./organised_files:/app/organised_files \
  -v ./reports:/app/reports \
  -v ./processing_completed:/app/processing_completed \
  -v ./processing_failed:/app/processing_failed \
  --name document-organiser \
  ocr-document-organiser
```

Open `http://localhost:5000`, drop PDFs into `input_files/`, and click **Process**.

Get a Gemini API key at [aistudio.google.com](https://aistudio.google.com/app/apikey) — the free tier is sufficient for typical document volumes.

---

## Local Development

<details>
<summary>Expand for non-Docker setup</summary>

**Prerequisites:**

- Python 3.12+
- Tesseract OCR
  - **Windows:** Installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). Default path: `C:\Program Files\Tesseract-OCR`
  - **macOS:** `brew install tesseract`
  - **Linux:** `sudo apt install tesseract-ocr`
- Poppler
  - **Windows:** See [this guide](https://stackoverflow.com/questions/18381713/how-to-install-poppler-on-windows); add to `PATH`
  - **macOS:** `brew install poppler`
  - **Linux:** `sudo apt install poppler-utils`

**Setup:**

```bash
git clone https://github.com/your-username/ocr-document-organiser.git
cd ocr-document-organiser

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

echo 'GEMINI_API_KEY="your_key_here"' > .env
```

On Windows, if Tesseract is not on your `PATH`, `main.py` already handles this — it auto-detects `os.name == 'nt'` and sets the executable path to the default install location. Update that path in `main.py` if you installed to a custom directory.

**Run:**

```bash
python main.py
```

Open `http://127.0.0.1:5000`.

</details>

---

## Folder Structure

```
ocr-document-organiser/
├── main.py                  # Flask app, OCR pipeline, SSE streaming
├── templates/
│   └── index.html           # Web UI
├── requirements.txt
├── Dockerfile
├── input_files/             # Drop PDFs here (created on first run)
├── organised_files/         # Renamed output files land here
├── processing_completed/    # Originals that processed successfully
├── processing_failed/       # Originals that failed — review manually
└── reports/                 # Timestamped CSV logs of each run
```

---

## Licence

MIT. See `LICENSE` for details.

---

## Contributing

Issues and pull requests are welcome. For significant changes, open an issue first to discuss the approach.
