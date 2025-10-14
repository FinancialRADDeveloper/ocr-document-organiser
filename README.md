# AI Document Organiser
This project uses AI to automatically rename and organise your scanned PDF documents. It performs local OCR to extract text from your files and then uses the Gemini AI to generate a consistent, descriptive filename based on the document's content.
A web interface is provided to show the results of the processing.
## Features
- **Local OCR:** Extracts text from PDFs on your machine using Tesseract to save on API costs.
- **AI-Powered Renaming:** Leverages a generative AI model to create clean, structured filenames.
- **Automated File Archiving:** Moves original files to `processed_files` or `failed_process_files` folders after each run.
- **Web Interface:** Displays a summary of the original and new filenames in your browser.
- **Dockerized:** Includes a Docker setup for easy, consistent, and cross-platform deployment.

## Folder Structure
- `input_files/`: Place your PDF files here for processing.
- `organised_files/`: Your renamed and organised PDF files will appear here.
- `processed_files/`: Original PDFs that were successfully processed are moved here.
- `failed_process_files/`: Original PDFs that failed to process are moved here for review.
- `reports/`: Contains CSV logs of the file processing operations.

## Setup and Usage
There are two methods to run the application. Using Docker is highly recommended as it handles all system dependencies automatically.
### Method 1: Running with Docker (Recommended)
This is the simplest way to get started.
#### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running on your system.

#### Configuration
1. **Clone the Repository:**
``` bash
    git clone <your-repository-url>
    cd ocr-document-organiser
```
1. **Create Environment File:** Create a file named in the project root and add your Gemini API key: `.env`
``` 
    # .env
    GEMINI_API_KEY="YOUR_API_KEY_HERE"
```
Replace `"YOUR_API_KEY_HERE"` with your actual key.
#### Build and Run
1. **Build the Docker Image:** Open a terminal in the project's root directory and run:
``` bash
    docker build -t ocr-document-organiser .
```
1. **Run the Docker Container:** Execute the following command to start the application. This command links your local folders to the container and securely passes your API key.
``` bash
    docker run -d \
      -e GEMINI_API_KEY=$(cat .env | grep GEMINI_API_KEY | cut -d '=' -f2) \
      -p 5000:5000 \
      -v ./input_files:/app/input_files \
      -v ./organised_files:/app/organised_files \
      -v ./reports:/app/reports \
      -v ./processed_files:/app/processed_files \
      -v ./failed_process_files:/app/failed_process_files \
      --name document-organiser-app \
      ocr-document-organiser
```
#### How to Use
1. **Add Files:** Place PDF documents into the `input_files` folder on your local machine.
2. **View Results:** Open your web browser and navigate to `http://localhost:5000`.
3. **Check Output:** The renamed files will appear in `organised_files`, and the original files will be sorted into `processed_files` or `failed_process_files`.

### Method 2: Running Locally with PyCharm (Advanced)
This method requires manual installation of system dependencies.
#### Prerequisites
- Python 3.12 or newer.
- PyCharm IDE.
- **Tesseract-OCR Engine:**
    - **Windows:** Download and run the installer from the [Tesseract at UB Mannheim page](https://github.com/UB-Mannheim/tesseract/wiki). Note the installation path (e.g., `C:\Program Files\Tesseract-OCR`).
    - **macOS:** `brew install tesseract`
    - **Linux:** `sudo apt update && sudo apt install tesseract-ocr`

- **Poppler Utility:**
    - **Windows:** Follow a guide to [install Poppler for Windows](https://stackoverflow.com/questions/18381713/how-to-install-poppler-on-windows) and add it to your system's PATH.
    - **macOS:** `brew install poppler`
    - **Linux:** `sudo apt install poppler-utils`

#### Configuration
1. **Clone the Repository** and open it as a project in PyCharm.
2. **Create a Virtual Environment** and install dependencies from . PyCharm will likely prompt you to do this automatically. `requirements.txt`
3. **Create file`.env`** as described in the Docker method.
4. **(Windows Only)** If you installed Tesseract to a custom location, you may need to uncomment and update the path in : `main.py`
``` python
    # main.py
    # ...
    # If Tesseract is not in your PATH, you may need to specify its location
    # Example for Windows:
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # ...
```
#### How to Use
1. **Run the Application:** Right-click the file in PyCharm and select "Run 'main'". `main.py`
2. **Add Files** and **View Results** as described in the Docker usage section.
