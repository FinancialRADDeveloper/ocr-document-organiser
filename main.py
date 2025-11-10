import os
import shutil
import re
from pathlib import Path

import pandas as pd
import pytesseract
from flask import Flask, render_template, request, Response
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
        You are an expert file organization assistant. Based on the document text provided,
        generate a concise and descriptive filename.

        The format should be: YYYY-MM-DD - Company - Document_Type - Subject - Reference_Number.pdf

        - Use the document's main date for YYYY-MM-DD. If no date is found, use 'Undated'.
        - Extract the primary company name.
        - Briefly describe the document type (e.g., 'Annuity Statement', 'Invoice', 'Insurance Policy').
        - Include a short subject or name (e.g., 'J Hunt').
        - Add a unique reference or policy number if available.
        - If a component is not available in the document, omit it from the filename.
        - Ensure the final filename is valid for Windows and macOS (no invalid characters like /\\:*?"<>|).
        - Do not add any extra explanation or text. Only return the suggested filename.
        """

        # Generate content from the extracted text
        response = model.generate_content([prompt, document_text])

        # Clean up the response to ensure it's a valid filename
        clean_name = re.sub(r'[\\/*?:"<>|]', "", response.text.strip())

        # Ensure it has .pdf extension
        if not clean_name.lower().endswith('.pdf'):
            clean_name += '.pdf'

        logger.info(f"  -> Suggested filename: {clean_name}")
        yield f"  -> Suggested filename: {clean_name}"
        return clean_name

    except Exception as e:
        error_msg = f"  -> An error occurred with the AI model: {e}"
        logger.error(error_msg)
        yield error_msg
        return None


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
    """Processes all PDF files in the input folder and yields progress."""

    def log_and_stream(message, level=logging.INFO):
        """Logs to console and yields for SSE stream."""
        logger.log(level, message)
        print(message, file=sys.stderr, flush=True)  # Print to stderr to avoid Flask redirection
        return f"data: {message}\n\n"

    def run_sub_process(generator):
        """Consumes a generator, logs and yields its messages, and returns its final value."""
        while True:
            try:
                message = next(generator)
                yield log_and_stream(message)
            except StopIteration as e:
                return e.value

    results_for_web = []
    results_for_csv = []

    try:
        # Ensure folders exist
        INPUT_FOLDER.mkdir(exist_ok=True)
        OUTPUT_FOLDER.mkdir(exist_ok=True)

        yield log_and_stream("Starting file processing...")

        # Iterate through files in the input folder
        for original_path in INPUT_FOLDER.glob("*.pdf"):
            yield log_and_stream(f"Processing: {original_path.name}")

            # Step 1: Extract text from the PDF
            document_text = yield from run_sub_process(extract_text_from_pdf_local(original_path))

            if not document_text:
                yield from run_sub_process(archive_original_file(original_path, success=False))
                continue

            # Step 2: Generate a new filename from the extracted text
            suggested_name = yield from run_sub_process(generate_filename_from_text(document_text))

            if not suggested_name:
                yield log_and_stream(f"  -> Could not generate a name for {original_path.name}. Skipping.", level=logging.WARNING)
                yield from run_sub_process(archive_original_file(original_path, success=False))
                continue

            # Copy and rename the file
            new_path = OUTPUT_FOLDER / suggested_name
            shutil.copy(original_path, new_path)

            yield log_and_stream(f"  -> Renamed and copied to: {new_path.name}")

            # Store result for web display
            results_for_web.append({
                'original_name': original_path.name,
                'new_name': suggested_name,
            })
            # Store results for the CSV report
            results_for_csv.append({
                'original_name': original_path.name,
                'ocr_text': document_text,
                'new_name': suggested_name,
            })

            # Archive the original file
            yield from run_sub_process(archive_original_file(original_path, success=True))

        yield log_and_stream("File processing complete.")

        # Save results to CSV file
        if results_for_csv:
            csv_path = save_results_to_csv(results_for_csv)
            if csv_path:
                yield log_and_stream(f"Final report is available at: {csv_path}")
            else:
                yield log_and_stream("Error: Failed to save the processing report.", level=logging.ERROR)

        # Use a specific event to send the final data and end the stream
        json_payload = json.dumps(results_for_web)
        yield f"event: end\ndata: {json_payload}\n\n"

    except Exception as e:
        # Catch any unexpected errors during processing
        error_message = "An unexpected error occurred during processing. Check console for details."
        # Log the full exception and traceback to the console
        logger.error(f"An unexpected error occurred during processing: {e}", exc_info=True)
        print(f"ERROR: An unexpected error occurred during processing: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        yield log_and_stream(error_message, level=logging.ERROR)

        # Still send an 'end' event so the client knows processing has stopped
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


if __name__ == '__main__':
    logger.info("--- Starting Web Server ---")
    logger.info("Open your browser and go to http://127.0.0.1:5000")
    print("--- Starting Web Server ---", file=sys.stderr, flush=True)
    print("Open your browser and go to http://127.0.0.1:5000", file=sys.stderr, flush=True)
    # Enable Flask debug only if FLASK_DEBUG env var is set, to avoid interfering with IDE debuggers
    use_debug = os.getenv('FLASK_DEBUG', '0') in ('1', 'true', 'True', 'yes')
    app.run(debug=use_debug, host='0.0.0.0', use_reloader=False)
