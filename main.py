import os
import shutil
import re
from pathlib import Path

import pandas as pd
import pytesseract
from flask import Flask, render_template, request, Response
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd
import json
from datetime import datetime
from pdf2image import convert_from_path
import logging

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

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
FINAL_DOCUMENTS_ROOT = Path("final_documents")


# --- AI Model Interaction ---
def list_gemini_models():
    """Lists available Gemini models for debugging."""
    print("\n--- Available Gemini Models ---")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
    except Exception as e:
        print(f"Could not list models: {e}")
    print("---------------------------------\n")


# --- AI Model Interaction ---
def extract_text_from_pdf(pdf_path):
    """Uploads a PDF to Gemini, extracts text, and deletes the file."""
    print(f"  -> Uploading and extracting text from {pdf_path.name}...")
    uploaded_file = None
    try:
        # Use a model that can handle PDF input, taken from your available list
        model = genai.GenerativeModel('gemini-2.0-flash')

        # Upload the file and prompt for text extraction
        uploaded_file = genai.upload_file(path=pdf_path)
        response = model.generate_content(["Extract all text from this document.", uploaded_file])

        print("  -> Text extraction complete.")
        return response.text
    except Exception as e:
        print(f"  -> An error occurred during text extraction: {e}")
        return None
    finally:
        # Clean up the file from the server
        if uploaded_file:
            uploaded_file.delete()
            print("  -> Temporary file deleted.")


def extract_text_from_pdf_local(pdf_path):
    """Extracts text from a PDF locally using Tesseract OCR."""
    yield f"  -> Extracting text locally from {pdf_path.name}..."
    try:
        # Note: pdf2image requires the poppler utility to be installed and in your PATH.
        images = convert_from_path(pdf_path)

        full_text = ""
        for i, image in enumerate(images):
            yield f"    -> Processing page {i + 1}/{len(images)}"
            full_text += pytesseract.image_to_string(image) + "\n"

        yield "  -> Local text extraction complete."
        return full_text
    except Exception as e:
        yield f"  -> An error occurred during local text extraction: {e}"
        yield "  -> Please ensure Tesseract OCR and poppler are installed and configured correctly."
        return None


def generate_filename_from_text(document_text):
    """Generates a new filename from the document's text content."""
    if not document_text:
        return

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

        yield f"  -> Suggested filename: {clean_name}"
        return clean_name

    except Exception as e:
        yield f"  -> An error occurred with the AI model: {e}"
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
        yield f"  -> Moved original file to {status} folder."
    except Exception as e:
        yield f"  -> Error moving original file: {e}"


# --- File Processing ---
def append_result_to_csv(result_data, csv_path):
    """Appends a single processing result to a CSV file."""
    try:
        # Create a DataFrame for the single result
        df = pd.DataFrame([result_data])

        # Ensure the columns are in the correct order for consistency
        df = df[['original_name', 'ocr_text', 'new_name']]

        # If the file doesn't exist yet, write the header
        header = not csv_path.exists()

        # Append to the CSV file
        df.to_csv(csv_path, mode='a', header=header, index=False, encoding='utf-8')

    except Exception as e:
        # Use logging for better error tracking
        logging.error(f"Error appending to CSV file: {e}")


def process_files():
    """Processes all PDF files in the input folder and yields progress."""

    def log_and_stream(message):
        """Logs to console and yields for SSE stream."""
        logging.info(message)
        return f"data: {message}\n\n"

    def run_sub_process(generator):
        """Consumes a generator, logs and yields its messages, and returns its final value."""
        while True:
            try:
                message = next(generator)
                yield log_and_stream(message)
            except StopIteration as e:
                return e.value

    # Ensure folders exist
    INPUT_FOLDER.mkdir(exist_ok=True)
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    results_for_web = []

    yield log_and_stream("Starting file processing...")

    # Generate a single timestamped CSV for this processing run
    REPORTS_FOLDER.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = REPORTS_FOLDER / f"processing_report_{timestamp}.csv"
    yield log_and_stream(f"Creating report at: {csv_path}")

    # list the models once for debugging
    # list_gemini_models()

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
            yield log_and_stream(f"  -> Could not generate a name for {original_path.name}. Skipping.")
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
        # Create a dictionary with the full results for the CSV
        result_for_csv = {
            'original_name': original_path.name,
            'ocr_text': document_text,
            'new_name': suggested_name,
        }

        # Append the result to the CSV file immediately after processing
        append_result_to_csv(result_for_csv, csv_path)

        # Archive the original file
        yield from run_sub_process(archive_original_file(original_path, success=True))

    yield log_and_stream("File processing complete.")

    # The CSV is now saved incrementally, but we can log the final path.
    yield log_and_stream(f"Final report is available at: {csv_path}")

    # Use a specific event to send the final data
    yield f"event: end\\ndata: {json.dumps(results_for_web)}\\n\\n"


# --- Secondary File Organization ---
def get_folder_suggestions(report_df, folder_structure):
    """
    Gets folder suggestions from Gemini for a DataFrame of files.

    Args:
        report_df (pd.DataFrame): DataFrame with file info.
        folder_structure (str): A string describing the target folder structure.

    Returns:
        pd.DataFrame: The DataFrame updated with a 'suggested_folder' column.
    """
    logging.info("Getting folder suggestions from Gemini...")
    try:
        model = genai.GenerativeModel('gemini-pro-latest')

        # Create a detailed prompt for the AI
        prompt_parts = [
            "You are an expert file organization assistant.",
            "You will be given a target folder structure and a list of files with their OCR text.",
            "Your task is to determine the best subfolder for each file within the given structure.",
            f"The root folder is '{FINAL_DOCUMENTS_ROOT.name}'. All suggestions must be a sub-path within this root.",
            f"Target Folder Structure:\n---\n{folder_structure}\n---\n",
            "Based on the file's content and name, decide the most appropriate folder for it.",
            "Return a single JSON object where keys are the 'new_name' of each file, and values are the suggested subfolder path (e.g., 'Statements/Bank', 'Invoices', etc.).",
            "If no folder is suitable, return an empty string for that file's value.",
            "Only return the JSON object, with no extra explanation or markdown.",
            "\nFile Data (JSON):\n",
            report_df[['original_name', 'ocr_text', 'new_name']].to_json(orient='records')
        ]

        response = model.generate_content(prompt_parts)
        # Clean the response to ensure it's valid JSON
        cleaned_response_text = response.text.strip().replace('`', '').replace('json', '')
        suggestions = json.loads(cleaned_response_text)

        # Map the suggestions back to the DataFrame
        report_df['suggested_folder'] = report_df['new_name'].map(suggestions)
        logging.info("Successfully received and mapped folder suggestions.")
        return report_df

    except Exception as e:
        logging.error(f"An error occurred while getting folder suggestions: {e}")
        return None


def move_files_to_folders(report_df, source_folder):
    """
    Moves files from the source folder to the suggested subfolders.

    Args:
        report_df (pd.DataFrame): DataFrame with 'new_name' and 'suggested_folder'.
        source_folder (Path): The folder where the renamed files currently are.
    """
    logging.info("Moving files to their final destination...")
    for index, row in report_df.iterrows():
        new_name = row.get('new_name')
        suggested_folder_str = row.get('suggested_folder')

        if not new_name or not suggested_folder_str:
            logging.warning(f"Skipping row due to missing 'new_name' or 'suggested_folder': {row}")
            continue

        source_path = source_folder / new_name
        destination_folder = FINAL_DOCUMENTS_ROOT / suggested_folder_str
        destination_path = destination_folder / new_name

        if source_path.exists():
            try:
                # For Windows, handle potential long paths by using an absolute path
                # with the special `\\?\` prefix. This allows paths longer than 260 chars.
                if os.name == 'nt':
                    abs_destination_folder = destination_folder.resolve()
                    # `os.makedirs` and `shutil.move` work with this prefix.
                    long_path_folder = "\\\\?\\" + str(abs_destination_folder)
                    long_path_file = "\\\\?\\" + str(destination_path.resolve())

                    os.makedirs(long_path_folder, exist_ok=True)
                    shutil.move(str(source_path), long_path_file)
                else:
                    # On other systems, the standard approach is fine.
                    destination_folder.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source_path), str(destination_path))

                logging.info(f"Moved '{source_path}' to '{destination_path}'")
            except Exception as e:
                logging.error(f"Could not move file '{source_path}': {e}")
        else:
            logging.warning(f"File '{source_path}' not found, skipping move.")


# --- Web Server ---
app = Flask(__name__)


@app.route('/')
def index():
    """Renders the main page with the uploader."""
    reports = sorted([f.name for f in REPORTS_FOLDER.glob('*.csv')], reverse=True)
    return render_template('index.html', reports=reports)


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


@app.route('/scan_structure')
def scan_structure_route():
    """Scans the final_documents directory and returns its structure."""
    FINAL_DOCUMENTS_ROOT.mkdir(exist_ok=True)
    # This is a simplified tree generator
    tree_lines = []
    for root, dirs, files in os.walk(FINAL_DOCUMENTS_ROOT):
        level = root.replace(str(FINAL_DOCUMENTS_ROOT), '').count(os.sep)
        if level >= 4: # Limit depth to 4 as requested
            dirs[:] = [] # Stop descending further
            continue
        indent = ' ' * 4 * level
        tree_lines.append(f'{indent}{os.path.basename(root)}/')
    return Response('\n'.join(tree_lines), mimetype='text/plain')


@app.route('/organize', methods=['POST'])
def organize_route():
    """Triggers the final organization of files based on a report."""
    report_filename = request.form.get('report')
    folder_structure = request.form.get('folder_structure')
    trial_run = request.form.get('trial_run') == 'true'

    if not report_filename or not folder_structure:
        return "Missing report filename or folder structure.", 400

    report_path = REPORTS_FOLDER / report_filename
    if not report_path.exists():
        return f"Report '{report_filename}' not found.", 404

    try:
        report_df = pd.read_csv(report_path)

        # Get folder suggestions from AI
        updated_df = get_folder_suggestions(report_df, folder_structure)
        if updated_df is None:
            return "Failed to get folder suggestions from the AI.", 500

        if trial_run:
            # For a trial run, just generate the plan and return it
            plan = []
            for _, row in updated_df.iterrows():
                if pd.notna(row['suggested_folder']) and row['suggested_folder']:
                    plan.append(f"MOVE '{row['new_name']}'\n  TO '{FINAL_DOCUMENTS_ROOT / row['suggested_folder']}'")
                else:
                    plan.append(f"SKIP '{row['new_name']}' (No folder suggested)")
            return Response('\n\n'.join(plan), mimetype='text/plain')
        else:
            # For a real run, save the suggestions and move the files
            updated_df.to_csv(report_path, index=False)
            logging.info(f"Updated report '{report_filename}' with folder suggestions.")
            move_files_to_folders(updated_df, OUTPUT_FOLDER)
            return "Organization process complete! Files have been moved.", 200

    except Exception as e:
        logging.error(f"An error occurred during organization: {e}")
        return f"An error occurred: {e}", 500


if __name__ == '__main__':
    print("\n--- Starting Web Server ---")
    print("Open your browser and go to http://127.0.0.1:5000")
    # Disabling the reloader to prevent silent exit issues.
    # This is often necessary if unused .py files confuse the debugger.
    app.run(debug=True, host='0.0.0.0', use_reloader=False)