# QUIRE: Smart PDF Editor and Analyzer

**QUIRE** is an all-in-one, privacy-focused digital document workspace built for students, researchers, and professionals. It combines essential PDF file utilities (merging, splitting, format conversion) with cognitive machine learning and Natural Language Processing (NLP) capabilities (extractive summarization, TF-IDF keyword extraction, and cosine similarity matching) inside a clean, responsive web interface.

---

## 🚀 Features

### 1. Document Utilities
* **PDF Merger:** Combine multiple PDF files into a single, cohesive document.
* **PDF Splitter:** Extract specific page ranges or individual pages from large PDF files.
* **Format Converters:** 
  * *Image-to-PDF:* Convert PNG, JPG, or JPEG images into standardized PDF layouts.
  * *Word-to-PDF:* Parse Microsoft Word (`.docx`) files and compile them into formatted PDFs.
* **Text Extractor:** Instantly read and extract raw text streams from searchable PDF documents.

### 2. Machine Learning & NLP Analytics
* **TF-IDF Keyword Extractor:** Tokenize document text, filter out stop words, and automatically rank the top domain keywords.
* **Extractive Text Summarizer:** Evaluate sentence position and frequency scoring to generate concise summaries of long reports.
* **Document Similarity (Cosine Similarity):** Compute vector space overlap between documents to quantify semantic textual similarity.

---

## 🛠️ Technology Stack

* **Frontend & UI Framework:** [Streamlit](https://streamlit.io/) (`streamlit`)
* **PDF Processing Engine:** [PyMuPDF](https://pymupdf.readthedocs.io/) (`fitz`)
* **Machine Learning & NLP:** [Scikit-learn](https://scikit-learn.org/) (`scikit-learn`), NumPy (`numpy`)
* **Document & Image Parsing:** `python-docx`, Pillow (`PIL`)
* **Data Handling:** Pandas (`pandas`)

---

## ⚙️ Installation & Setup Instructions

Follow these steps to set up and run **QUIRE** locally on your machine:

### Prerequisites
Ensure you have **Python 3.9 – 3.12** installed on your operating system.

### Step 1: Clone or Download the Repository
Extract your project folder and navigate into the root directory via your terminal or command prompt:
```bash
cd quire-pdf-editor
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
python -m venv venv
```
* **Activate the environment:**
  * *Windows:* `venv\Scripts\activate`
  * *macOS / Linux:* `source venv/bin/activate`

### Step 3: Install Dependencies
Install all required Python packages using `pip`:
```bash
pip install streamlit pymupdf scikit-learn pandas python-docx pillow numpy
```

---

## ▶️ Running the Application

Once all dependencies are installed, launch the Streamlit web server by running:
```bash
streamlit run app.py
```

Streamlit will start a local development server and automatically open the dashboard in your default web browser at:
* **Local URL:** `http://localhost:8501`

---

## 📂 Project Directory Structure

```text
QUIRE/
│
├── app.py                  # Main Streamlit application script
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation & instructions
└── assets/                 # (Optional) Icons, screenshots, or test files
```

---

## 🔒 Privacy & Security Note
QUIRE executes file processing locally on your machine's runtime environment. Uploaded documents and raw text streams are handled in-memory and are never uploaded or exposed to third-party external servers.

---
## 📄 License
This project is developed as an open-source BCA mini-project under academic guidelines.