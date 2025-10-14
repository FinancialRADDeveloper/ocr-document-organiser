import os
import shutil
import re
from pathlib import Path
from flask import Flask, render_template
import google.generativeai as genai
from dotenv import load_dotenv
import pypdf

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Define folder paths
INPUT_FOLDER = Path("input_files")
OUTPUT_FOLDER = Path("organised_files")


# --- AI Model Interaction ---
def get_new_filename(file_content, original_extension):
    """Generates a new filename using the Gemini model."""

    # Configure the generative model
    model = genai.GenerativeModel('gemini-1.5-flash')

    prompt = f"""
    You are an expert file organization assistant. Based on the following document text,
    generate a concise and descriptive filename.

    The format should be: YYYY-MM-DD - Company - Document_Type - Subject - Reference_Number{original_extension}

    - Use the document's main date for YYYY-MM-DD. If no date is found, use 'Undated'.
    - Extract the primary company name.
    - Briefly describe the document type (e.g., 'Annuity Statement', 'Invoice', 'Insurance Policy').
    - Include a short subject or name (e.g., 'J Hunt').
    - Add a unique reference or policy number if available.
    - If a component is not available in the text, omit it from the filename.
    - Ensure the final filename is valid for Windows and macOS (no invalid characters like /\\:*?"<>|).
    - Do not add any extra explanation or text. Only return the suggested filename.

    DOCUMENT TEXT:
    ---
    {file_content[:4000]}
    ---
    """

    try:
        response = model.generate_content(prompt)
        # Clean up the response to ensure it's a valid filename
        clean_name = re.sub(r'[\\/*?:"<>|]', "", response.text.strip())
        return clean_name
    except Exception as e:
        print(f"An error occurred with the AI model: {e}")
        return None


# --- File Processing ---
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        with open(pdf_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Could not read PDF {pdf_path}: {e}")
        return None


def process_files():
    """Processes all PDF files in the input folder."""

    # Ensure folders exist
    INPUT_FOLDER.mkdir(exist_ok=True)
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    results = []

    print("Starting file processing...")

    # Iterate through files in the input folder
    for original_path in INPUT_FOLDER.glob("*.pdf"):
        print(f"Processing: {original_path.name}")

        # 1. Extract text
        file_text = extract_text_from_pdf(original_path)
        if not file_text:
            continue

        # 2. Get new filename from AI
        suggested_name = get_new_filename(file_text, original_path.suffix)
        if not suggested_name:
            print(f"  -> Could not generate a name for {original_path.name}. Skipping.")
            continue

        # 3. Copy and rename the file
        new_path = OUTPUT_FOLDER / suggested_name
        shutil.copy(original_path, new_path)

        print(f"  -> Renamed and copied to: {new_path.name}")

        # 4. Store result for web display
        results.append({
            'original_name': original_path.name,
            'original_path': str(original_path.resolve()),
            'new_name': new_path.name,
            'new_path': str(new_path.resolve())
        })

    print("File processing complete.")
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
    app.run(debug=True)
