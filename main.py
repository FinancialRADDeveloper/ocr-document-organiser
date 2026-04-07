import os
import shutil
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pytesseract
from flask import Flask, render_template, request, Response, send_from_directory
import google.generativeai as genai
from dotenv import load_dotenv
import json
from datetime import datetime
from pdf2image import convert_from_path
import logging
import sys

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- Logging Configuration ---
# Note: Avoid forcing unbuffered stdout/stderr to keep IDE debuggers stable.
# If you need unbuffered logs, configure logging handlers instead of re-wrapping sys.stdout/sys.stderr.

# Remove all existing handlers and configure fresh
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logger with console handler pointing to stderr (which Flask doesn't redirect)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)

logger = logging.getLogger(__name__)

# --- Tesseract Configuration for Windows ---
# If running on Windows, specify the path to the Tesseract executable.
# This is necessary because the installer doesn't always add it to the system's PATH.
# On other systems (like Linux in Docker), Tesseract is expected to be in the PATH.
if os.name == 'nt':  # 'nt' is the name for Windows
    # Update this path if you installed Tesseract in a different location
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define folder paths
INPUT_FOLDER = Path("input_files")
OUTPUT_FOLDER = Path("organised_files")
REPORTS_FOLDER = Path("reports")
PROCESSED_FOLDER = Path("processing_completed")
FAILED_FOLDER = Path("processing_failed")

# Parallelism settings
MAX_WORKERS = 4

# Lock to make the duplicate-filename check + copy atomic across threads
_file_lock = threading.Lock()

# Gemini 2.0 Flash pricing (USD per 1M tokens)
GEMINI_PRICE_INPUT_PER_M  = 0.075
GEMINI_PRICE_OUTPUT_PER_M = 0.30


# --- AI Model Interaction ---
def list_gemini_models():
    """Lists available Gemini models for debugging."""
    logger.info("--- Available Gemini Models ---")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                logger.info(f"- {m.name}")
    except Exception as e:
        logger.error(f"Could not list models: {e}")
    logger.info("---------------------------------")


# --- AI Model Interaction ---
def extract_text_from_pdf(pdf_path):
    """Uploads a PDF to Gemini, extracts text, and deletes the file."""
    logger.info(f"  -> Uploading and extracting text from {pdf_path.name}...")
    uploaded_file = None
    try:
        # Use a model that can handle PDF input, taken from your available list
        model = genai.GenerativeModel('gemini-2.0-flash')

        # Upload the file and prompt for text extraction
        uploaded_file = genai.upload_file(path=pdf_path)
        response = model.generate_content(["Extract all text from this document.", uploaded_file])

        logger.info("  -> Text extraction complete.")
        return response.text
    except Exception as e:
        logger.error(f"  -> An error occurred during text extraction: {e}")
        return None
    finally:
        # Clean up the file from the server
        if uploaded_file:
            uploaded_file.delete()
            logger.info("  -> Temporary file deleted.")


def extract_text_from_pdf_local(pdf_path):
    """Extracts text from a PDF locally using Tesseract OCR."""
    logger.info(f"  -> Extracting text locally from {pdf_path.name}...")
    yield f"  -> Extracting text locally from {pdf_path.name}..."
    try:
        # Note: pdf2image requires the poppler utility to be installed and in your PATH.
        images = convert_from_path(pdf_path)

        full_text = ""
        for i, image in enumerate(images):
            msg = f"    -> Processing page {i + 1}/{len(images)}"
            logger.info(msg)
            yield msg
            full_text += pytesseract.image_to_string(image) + "\n"

        logger.info("  -> Local text extraction complete.")
        yield "  -> Local text extraction complete."
        return full_text
    except Exception as e:
        error_msg = f"  -> An error occurred during local text extraction: {e}"
        logger.error(error_msg)
        yield error_msg
        yield "  -> Please ensure Tesseract OCR and poppler are installed and configured correctly."
        return None


def generate_filename_from_text(document_text):
    """Generates a new filename from the document's text content."""
    if not document_text:
        return

    logger.info("  -> Generating new filename from text...")
    yield "  -> Generating new filename from text..."
    try:
        # Use a reasoning model, taken from your available list
        model = genai.GenerativeModel('gemini-pro-latest')

        prompt = """
You are an expert document management assistant specialising in organising scanned personal and
business documents. Your task is to analyse the provided document text and return a structured
JSON object describing the document.

## Output format

Return ONLY a valid JSON object — no markdown fences, no explanation, no extra text:

{
  "filename": "YYYY-MM-DD - Company - Document Type - Subject - REFnumber.pdf",
  "confidence": "high" | "medium" | "low",
  "document_type": "<one of the types listed below>"
}

## Filename construction rules

1. **Date** (YYYY-MM-DD): Use the primary document date — the date the document was issued or
   the statement period end date. If multiple dates appear, prefer the issue/effective date over
   printed/processing dates. If no date is found, use "Undated".

2. **Company**: Extract the issuing organisation's name (e.g. "HSBC", "HMRC", "NHS England",
   "British Gas"). If the header is a full postal address, extract only the company/organisation
   name from it. Do not include street addresses.

3. **Document Type**: Choose the single best match from the vocabulary below. Use "Document" only
   if nothing else fits.

4. **Subject** (optional): A brief identifier such as a person's name (e.g. "J Smith"), an
   account nickname, or a property address. Omit if not meaningful or not present.

5. **Reference** (optional): The most unique identifier on the document — policy number, invoice
   number, NI number (partial), tax reference, etc. Prefix with "REF" only if no natural label
   exists. Omit if not present.

6. Separate components with " - " (space-hyphen-space).
7. Omit any component that is not present or cannot be determined — do not use placeholders.
8. The filename MUST be safe for Windows and macOS: no characters from \\ / : * ? " < > |
9. Keep the filename under 200 characters total.
10. Always end with ".pdf".

## Document type vocabulary

Use exactly one of the following (or the closest match):
Bank Statement, Invoice, Receipt, Insurance Policy, Insurance Certificate, Insurance Schedule,
Payslip, Tax Return, Tax Notice, P60, P45, P11D, Self Assessment, VAT Return,
Council Tax Notice, Council Tax Bill, Utility Bill, Broadband Bill, Phone Bill, Energy Bill,
Water Bill, Mortgage Statement, Mortgage Offer, Tenancy Agreement, Lease Agreement,
Rental Statement, Pension Statement, Pension Letter, Annuity Statement, Investment Statement,
Savings Statement, Credit Card Statement, Loan Statement, Hire Purchase Agreement,
NHS Letter, NHS Appointment, Prescription, Medical Report, Hospital Discharge Summary,
Solicitor Letter, Legal Notice, Court Order, Grant Offer Letter, Bursary Letter,
Driving Licence, Passport Copy, Birth Certificate, Marriage Certificate, Death Certificate,
Employment Contract, Redundancy Notice, Reference Letter, DWP Letter, Universal Credit Letter,
Jury Summons, Planning Permission, Building Regulations, Warranty, User Manual,
Form, Correspondence, Document

## Edge case handling

- **Personal/sensitive documents** (medical, legal): still name them professionally using the
  rules above; do not redact or refuse.
- **Forms**: if the document is clearly a blank or partially filled form, use "Form" as the type.
- **Multiple companies**: use the issuing company, not any third party mentioned in the body.
- **Scanned quality**: if text is garbled, use best-effort extraction; set confidence to "low".
- **No recognisable content**: return filename "Undated - Unknown - Document.pdf",
  confidence "low", document_type "Document".

## Confidence levels

- "high": date, company, and document type all clearly identified
- "medium": one component is uncertain or missing
- "low": significant ambiguity or very little usable text
"""

        # Generate content from the extracted text
        response = model.generate_content([prompt, document_text])
        raw_text = response.text.strip()

        # Capture token usage and calculate cost
        usage = response.usage_metadata
        input_tokens  = usage.prompt_token_count     if usage else 0
        output_tokens = usage.candidates_token_count if usage else 0
        cost_usd = (input_tokens  / 1_000_000 * GEMINI_PRICE_INPUT_PER_M +
                    output_tokens / 1_000_000 * GEMINI_PRICE_OUTPUT_PER_M)

        # --- Parse JSON response ---
        clean_name = None
        document_type = None

        # Strip markdown code fences if the model wrapped the JSON
        fenced = re.match(r'^```(?:json)?\s*([\s\S]*?)\s*```$', raw_text, re.IGNORECASE)
        json_text = fenced.group(1) if fenced else raw_text

        try:
            parsed = json.loads(json_text)
            raw_filename = parsed.get("filename", "").strip()
            document_type = parsed.get("document_type", "").strip()
            confidence = parsed.get("confidence", "").strip()

            if raw_filename:
                # Sanitise the filename extracted from JSON
                clean_name = re.sub(r'[\\/*?:"<>|]', "", raw_filename)
                if not clean_name.lower().endswith('.pdf'):
                    clean_name += '.pdf'

                if document_type:
                    logger.info(f"  -> Document type: {document_type} (confidence: {confidence})")
                    yield f"  -> Document type: {document_type} (confidence: {confidence})"

        except (json.JSONDecodeError, AttributeError) as json_err:
            logger.warning(f"  -> JSON parsing failed ({json_err}); falling back to raw response text.")
            yield f"  -> JSON parsing failed; using raw model response as filename."

        # Fallback: treat the entire raw response as a filename string
        if not clean_name:
            clean_name = re.sub(r'[\\/*?:"<>|]', "", raw_text)
            if not clean_name.lower().endswith('.pdf'):
                clean_name += '.pdf'

        logger.info(f"  -> Suggested filename: {clean_name}")
        yield f"  -> Suggested filename: {clean_name}"

        token_msg = f"  -> Tokens used: {input_tokens:,} in / {output_tokens:,} out  |  Est. cost: ~${cost_usd:.5f}"
        logger.info(token_msg)
        yield token_msg

        return (clean_name, cost_usd, input_tokens, output_tokens)

    except Exception as e:
        error_msg = f"  -> An error occurred with the AI model: {e}"
        logger.error(error_msg)
        yield error_msg
        return (None, 0.0, 0, 0)


def archive_original_file(original_path, success):
    """Moves the original input file to a success or failure folder."""
    try:
        if success:
            destination_folder = PROCESSED_FOLDER
            status = "completed"
        else:
            destination_folder = FAILED_FOLDER
            status = "failed"

        # Ensure destination folder exists
        destination_folder.mkdir(exist_ok=True)

        # Move the file
        shutil.move(original_path, destination_folder / original_path.name)
        msg = f"  -> Moved original file to {status} folder."
        logger.info(msg)
        yield msg
    except Exception as e:
        error_msg = f"  -> Error moving original file: {e}"
        logger.error(error_msg)
        yield error_msg


# --- File Processing ---
def consume(gen):
    """Consume a generator, return (messages, return_value)."""
    messages = []
    try:
        while True:
            messages.append(next(gen))
    except StopIteration as e:
        return messages, e.value


def process_single_file(original_path):
    """
    Run the full pipeline for one PDF file in a thread.

    Returns a dict with keys:
        original_name  – str
        new_name       – str
        status         – "success" | "failed"
        log_lines      – list[str]
        cost_usd       – float  (placeholder; cost tracking added separately)
    """
    log_lines = []
    original_name = original_path.name

    try:
        log_lines.append(f"Processing: {original_name}")

        # Step 1: OCR
        ocr_msgs, document_text = consume(extract_text_from_pdf_local(original_path))
        log_lines.extend(ocr_msgs)

        if not document_text:
            log_lines.append(f"  -> OCR failed for {original_name}. Marking as failed.")
            archive_msgs, _ = consume(archive_original_file(original_path, success=False))
            log_lines.extend(archive_msgs)
            return {
                'original_name': original_name,
                'new_name': original_name,
                'status': 'failed',
                'log_lines': log_lines,
                'cost_usd': 0.0,
            }

        # Step 2: AI filename generation
        name_msgs, suggested_name = consume(generate_filename_from_text(document_text))
        log_lines.extend(name_msgs)

        if not suggested_name:
            log_lines.append(f"  -> Could not generate a name for {original_name}. Skipping.")
            archive_msgs, _ = consume(archive_original_file(original_path, success=False))
            log_lines.extend(archive_msgs)
            return {
                'original_name': original_name,
                'new_name': original_name,
                'status': 'failed',
                'log_lines': log_lines,
                'cost_usd': 0.0,
            }

        # Step 3: Copy to output (thread-safe duplicate check)
        with _file_lock:
            new_path = OUTPUT_FOLDER / suggested_name
            stem = new_path.stem
            suffix = new_path.suffix
            counter = 1
            while new_path.exists():
                new_path = OUTPUT_FOLDER / f"{stem} ({counter}){suffix}"
                counter += 1
            shutil.copy(original_path, new_path)

        final_name = new_path.name
        log_lines.append(f"  -> Renamed and copied to: {final_name}")

        # Step 4: Archive original
        archive_msgs, _ = consume(archive_original_file(original_path, success=True))
        log_lines.extend(archive_msgs)

        # Parse document type from filename segments
        name_parts = final_name.replace('.pdf', '').split(' - ')
        doc_type = name_parts[2].replace('_', ' ') if len(name_parts) >= 3 else 'Document'

        return {
            'original_name': original_name,
            'new_name': final_name,
            'document_type': doc_type,
            'ocr_text': document_text,
            'status': 'success',
            'log_lines': log_lines,
            'cost_usd': 0.0,
        }

    except Exception as exc:
        log_lines.append(f"  -> Unexpected error processing {original_name}: {exc}")
        logger.error(f"Unexpected error processing {original_name}: {exc}", exc_info=True)
        # Best-effort archive to failed folder
        try:
            archive_msgs, _ = consume(archive_original_file(original_path, success=False))
            log_lines.extend(archive_msgs)
        except Exception:
            pass
        return {
            'original_name': original_name,
            'new_name': original_name,
            'status': 'failed',
            'log_lines': log_lines,
            'cost_usd': 0.0,
        }


def save_results_to_csv(results_data):
    """Saves processing results to a timestamped CSV file using pandas."""
    if not results_data:
        logger.info("No results to save.")
        print("No results to save.", file=sys.stderr, flush=True)
        return None

    # Ensure reports folder exists
    REPORTS_FOLDER.mkdir(exist_ok=True)

    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = REPORTS_FOLDER / f"processing_report_{timestamp}.csv"

    try:
        # Create a DataFrame from the results
        df = pd.DataFrame(results_data)

        # Define the columns for the CSV
        df = df[['original_name', 'ocr_text', 'new_name']]

        # Save to CSV
        df.to_csv(csv_filename, index=False, encoding='utf-8')

        logger.info(f"Results saved to: {csv_filename}")
        print(f"Results saved to: {csv_filename}", file=sys.stderr, flush=True)
        return str(csv_filename)
    except Exception as e:
        logger.error(f"Error saving CSV file: {e}")
        print(f"Error saving CSV file: {e}", file=sys.stderr, flush=True)
        return None


def process_files():
    """Processes all PDF files in the input folder concurrently and yields SSE progress."""

    def log_and_stream(message, level=logging.INFO):
        """Logs to console and returns an SSE data line."""
        logger.log(level, message)
        print(message, file=sys.stderr, flush=True)
        return f"data: {message}\n\n"

    results_for_web = []
    results_for_csv = []
    session_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    try:
        # Ensure folders exist
        INPUT_FOLDER.mkdir(exist_ok=True)
        OUTPUT_FOLDER.mkdir(exist_ok=True)

        yield log_and_stream("Starting file processing...")

        pdf_paths = list(INPUT_FOLDER.glob("*.pdf"))

        if not pdf_paths:
            yield log_and_stream("No PDF files found in input folder.")
        else:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit all files up-front
                future_to_path = {
                    executor.submit(process_single_file, path): path
                    for path in pdf_paths
                }

                # Stream results as each file completes (fastest first)
                for future in as_completed(future_to_path):
                    result = future.result()

                    # Replay all log lines from the worker thread into the SSE stream
                    for line in result['log_lines']:
                        yield log_and_stream(line)
                    session_cost += result.get('cost_usd', 0.0)

                    # Build web / CSV records from the result dict
                    web_record = {
                        'original_name': result['original_name'],
                        'new_name': result['new_name'],
                        'document_type': result.get('document_type', 'Unknown'),
                        'status': result['status'],
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    }
                    results_for_web.append(web_record)

                    if result['status'] == 'success':
                        results_for_csv.append({
                            'original_name': result['original_name'],
                            'ocr_text': result.get('ocr_text', ''),
                            'new_name': result['new_name'],
                        })

        yield log_and_stream("File processing complete.")

        # Save results to CSV file
        if results_for_csv:
            csv_path = save_results_to_csv(results_for_csv)
            if csv_path:
                yield log_and_stream(f"Final report is available at: {csv_path}")
            else:
                yield log_and_stream("Error: Failed to save the processing report.", level=logging.ERROR)

        # Send final SSE event with results and cost summary
        end_payload = {
            "results": results_for_web,
            "session_cost": round(session_cost, 5),
        }
        yield f"event: end\ndata: {json.dumps(end_payload)}\n\n"

    except Exception as e:
        error_message = "An unexpected error occurred during processing. Check console for details."
        logger.error(f"An unexpected error occurred during processing: {e}", exc_info=True)
        print(f"ERROR: An unexpected error occurred during processing: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        yield log_and_stream(error_message, level=logging.ERROR)

        error_payload = json.dumps({"error": error_message})
        yield f"event: end\ndata: {error_payload}\n\n"


# --- Web Server ---
app = Flask(__name__)


@app.route('/')
def index():
    """Renders the main page with the uploader and latest report."""
    REPORTS_FOLDER.mkdir(exist_ok=True)
    report_files = sorted(REPORTS_FOLDER.glob("*.csv"), key=os.path.getmtime, reverse=True)

    latest_report_data = []
    if report_files:
        latest_report_path = report_files[0]
        try:
            df = pd.read_csv(latest_report_path)
            # Truncate ocr_text for display
            if 'ocr_text' in df.columns:
                df['ocr_text'] = df['ocr_text'].astype(str).str.slice(0, 100) + '...'
            else:
                df['ocr_text'] = "N/A"

            # Ensure required columns exist, fill with placeholder if not
            for col in ['original_name', 'new_name', 'ocr_text']:
                if col not in df.columns:
                    df[col] = "N/A"

            # Select and reorder columns for display
            df = df[['original_name', 'new_name', 'ocr_text']]

            latest_report_data = df.to_dict(orient='records')
        except Exception as e:
            logger.error(f"Error reading or processing report file: {e}")
            print(f"Error reading or processing report file: {e}", file=sys.stderr, flush=True)

    return render_template('index.html', latest_report=latest_report_data)


@app.route('/upload', methods=['POST'])
def upload_route():
    """Handles file uploads."""
    INPUT_FOLDER.mkdir(exist_ok=True)

    if 'files[]' not in request.files:
        return 'No file part', 400

    files = request.files.getlist('files[]')

    for file in files:
        if file.filename and file.filename.lower().endswith('.pdf'):
            file.save(INPUT_FOLDER / file.filename)

    return 'Upload complete', 200


@app.route('/process')
def process_route():
    """Triggers the file processing and streams back the status."""
    return Response(process_files(), mimetype='text/event-stream')


# --- File & Organise routes ---

@app.route('/api/browse')
def api_browse():
    """Returns the contents of a directory as JSON for the folder tree browser."""
    from flask import jsonify
    path = request.args.get('path', '')
    if not path:
        if os.name == 'nt':
            import string
            drives = [f"{d}:\\" for d in string.ascii_uppercase if Path(f"{d}:\\").exists()]
            return jsonify({'path': '', 'dirs': [{'name': d, 'path': d} for d in drives]})
        else:
            return jsonify({'path': '/', 'dirs': [{'name': '/', 'path': '/'}]})

    target = Path(path)
    if not target.exists() or not target.is_dir():
        return jsonify({'error': 'Path not found'}), 404

    try:
        dirs = []
        for item in sorted(target.iterdir()):
            if item.is_dir() and not item.name.startswith('.'):
                try:
                    has_children = any(True for c in item.iterdir() if c.is_dir() and not c.name.startswith('.'))
                except PermissionError:
                    has_children = False
                dirs.append({'name': item.name, 'path': str(item), 'hasChildren': has_children})
        return jsonify({'path': str(target), 'dirs': dirs})
    except PermissionError:
        return jsonify({'error': 'Permission denied'}), 403


@app.route('/api/suggest-filing', methods=['POST'])
def api_suggest_filing():
    """Suggests filing destinations for PDFs in source_folder using Gemini."""
    from flask import jsonify
    data = request.get_json()
    source_folder = Path(data.get('source_folder', ''))
    destination_root = Path(data.get('destination_root', ''))

    if not source_folder.exists() or not destination_root.exists():
        return jsonify({'error': 'Invalid paths'}), 400

    # Get destination folder tree (2 levels deep)
    dest_folders = []
    for item in destination_root.rglob('*'):
        if item.is_dir() and not item.name.startswith('.'):
            depth = len(item.relative_to(destination_root).parts)
            if depth <= 2:
                dest_folders.append(str(item))

    # For each PDF in source, suggest destination
    suggestions = []
    model = genai.GenerativeModel('gemini-2.0-flash')

    for pdf in sorted(source_folder.glob('*.pdf')):
        try:
            prompt = f"""You are a document filing assistant.
Document filename: {pdf.name}
Available destination folders:
{chr(10).join(dest_folders)}

Based on the filename alone, suggest the single most appropriate destination folder from the list above.
Return ONLY the exact folder path from the list. Nothing else. No explanation."""

            response = model.generate_content(prompt)
            suggested = response.text.strip()

            # Validate the suggestion is actually in our list
            if suggested not in dest_folders:
                suggested = str(destination_root)  # fallback to root

            suggestions.append({
                'filename': pdf.name,
                'source_path': str(pdf),
                'suggested_destination': suggested,
                'destination_folder': Path(suggested).name,
            })
        except Exception as e:
            suggestions.append({
                'filename': pdf.name,
                'source_path': str(pdf),
                'suggested_destination': str(destination_root),
                'destination_folder': '(root)',
                'error': str(e),
            })

    return jsonify({'suggestions': suggestions})


@app.route('/api/apply-filing', methods=['POST'])
def api_apply_filing():
    """Moves files to their suggested destinations and archives originals."""
    from flask import jsonify
    data = request.get_json()
    moves = data.get('moves', [])

    FILED_FOLDER = Path('successfully_filed')
    FILED_FOLDER.mkdir(exist_ok=True)

    results = []
    for move in moves:
        source = Path(move['source_path'])
        destination_dir = Path(move['destination_folder'])

        try:
            destination_dir.mkdir(parents=True, exist_ok=True)
            dest_file = destination_dir / source.name

            # Handle duplicates
            if dest_file.exists():
                stem = source.stem
                suffix = source.suffix
                dest_file = destination_dir / f"{stem} ({datetime.now().strftime('%Y%m%d_%H%M%S')}){suffix}"

            shutil.copy2(source, dest_file)
            shutil.move(str(source), FILED_FOLDER / source.name)
            results.append({'filename': source.name, 'status': 'success', 'destination': str(dest_file)})
        except Exception as e:
            results.append({'filename': source.name, 'status': 'error', 'error': str(e)})

    return jsonify({'results': results})


@app.route('/files/<path:filename>')
def serve_organised_file(filename):
    """Serves a renamed PDF from the organised_files folder for inline viewing."""
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == '__main__':
    logger.info("--- Starting Web Server ---")
    logger.info("Open your browser and go to http://127.0.0.1:5000")
    print("--- Starting Web Server ---", file=sys.stderr, flush=True)
    print("Open your browser and go to http://127.0.0.1:5000", file=sys.stderr, flush=True)
    # Enable Flask debug only if FLASK_DEBUG env var is set, to avoid interfering with IDE debuggers
    use_debug = os.getenv('FLASK_DEBUG', '0') in ('1', 'true', 'True', 'yes')
    app.run(debug=use_debug, host='0.0.0.0', use_reloader=False)
