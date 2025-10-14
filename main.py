import os
import shutil
import re
from pathlib import Path

import pytesseract
from flask import Flask, render_template
import google.generativeai as genai
from dotenv import load_dotenv
import csv
from datetime import datetime
from pdf2image import convert_from_path


# --- Configuration ---
# Load environment variables from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Define folder paths
INPUT_FOLDER = Path("input_files")
OUTPUT_FOLDER = Path("organised_files")
REPORTS_FOLDER = Path("reports")

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
    print(f"  -> Extracting text locally from {pdf_path.name}...")
    try:
        # Note: pdf2image requires the poppler utility to be installed and in your PATH.
        images = convert_from_path(pdf_path)

        full_text = ""
        for i, image in enumerate(images):
            print(f"    -> Processing page {i+1}/{len(images)}")
            full_text += pytesseract.image_to_string(image) + "\n"

        print("  -> Local text extraction complete.")
        return full_text
    except Exception as e:
        print(f"  -> An error occurred during local text extraction: {e}")
        print("  -> Please ensure Tesseract OCR and poppler are installed and configured correctly.")
        return None



def generate_filename_from_text(document_text):
    """Generates a new filename from the document's text content."""
    if not document_text:
        return None

    print("  -> Generating new filename from text...")
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

        print(f"  -> Suggested filename: {clean_name}")
        return clean_name

    except Exception as e:
        print(f"  -> An error occurred with the AI model: {e}")
        return None


# --- File Processing ---
def save_results_to_csv(results):
    """Saves processing results to a timestamped CSV file."""

    # Ensure reports folder exists
    REPORTS_FOLDER.mkdir(exist_ok=True)

    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = REPORTS_FOLDER / f"processing_report_{timestamp}.csv"

    # Write results to CSV
    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'original_name', 'original_path', 'new_name', 'new_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in results:
                writer.writerow({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'original_name': result['original_name'],
                    'original_path': result['original_path'],
                    'new_name': result['new_name'],
                    'new_path': result['new_path']
                })

        print(f"\nResults saved to: {csv_filename}")
        return str(csv_filename)
    except Exception as e:
        print(f"Error saving CSV file: {e}")
        return None


def process_files():
    """Processes all PDF files in the input folder."""

    # Ensure folders exist
    INPUT_FOLDER.mkdir(exist_ok=True)
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    results = []

    print("Starting file processing...")

    # list the models once for debugging
    list_gemini_models()

    # Iterate through files in the input folder
    for original_path in INPUT_FOLDER.glob("*.pdf"):
        print(f"Processing: {original_path.name}")

        # Step 1: Extract text from the PDF
        document_text = extract_text_from_pdf_local(original_path)

        # Step 2: Generate a new filename from the extracted text
        suggested_name = generate_filename_from_text(document_text)

        if not suggested_name:
            print(f"  -> Could not generate a name for {original_path.name}. Skipping.")
            continue

        # Copy and rename the file
        new_path = OUTPUT_FOLDER / suggested_name
        shutil.copy(original_path, new_path)

        print(f"  -> Renamed and copied to: {new_path.name}")

        # Store result for web display
        results.append({
            'original_name': original_path.name,
            'original_path': str(original_path.resolve()),
            'new_name': new_path.name,
            'new_path': str(new_path.resolve())
        })

    print("File processing complete.")

    # Save results to CSV file
    if results:
        save_results_to_csv(results)

    return results


# --- Web Server ---
app = Flask(__name__)

# Process files on startup and store results globally
processed_results = process_files()


@app.route('/')
def show_results():
    """Renders the results page."""
    return render_template('results.html', results=processed_results)


if __name__ == '__main__':
    print("\n--- Starting Web Server ---")
    print("Open your browser and go to http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0')

