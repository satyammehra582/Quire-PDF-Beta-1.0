import io
import re
import fitz  # PyMuPDF
from PIL import Image
from docx import Document
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="QUIRE PDF- By Satyam", layout="wide")

# ==============================================================================
# CONVERSION FUNCTIONS
# ==============================================================================

def convert_images_to_pdf(image_list):
    """Converts a list of uploaded image files into a single PDF byte stream."""
    image_objects = []
    for img_file in image_list:
        img = Image.open(img_file)
        # Convert RGBA or Palette images to RGB to avoid PDF saving errors
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        image_objects.append(img)
    
    if not image_objects:
        return None
    
    output_stream = io.BytesIO()
    # Save the first image and append the rest
    image_objects[0].save(
        output_stream, "PDF", save_all=True, append_images=image_objects[1:]
    )
    return output_stream.getvalue()


def convert_docx_to_pdf_bytes(docx_file):
    """Extracts text from a Word document and writes it into a clean PDF."""
    doc = Document(docx_file)
    pdf_doc = fitz.open()
    page = pdf_doc.new_page() # Create a blank page
    
    # Simple text injection for Word document conversion
    y_position = 50
    for para in doc.paragraphs:
        if para.text.strip():
            if y_position > 750: # Create a new page if it overflows
                page = pdf_doc.new_page()
                y_position = 50
            page.insert_text((50, y_position), para.text, fontsize=11)
            y_position += 20
            
    output_stream = io.BytesIO()
    pdf_doc.save(output_stream)
    return output_stream.getvalue()

# [Keep your previous helper functions for extract_text_from_pdf, merge_pdfs, split_pdf, 
#  extract_keywords_tfidf, summarize_text_extractive, classify_document, and compute_pdf_similarity here]


# ==============================================================================
# HELPER FUNCTIONS: PDF UTILITIES (PyMuPDF)
# ==============================================================================

def extract_text_from_pdf(pdf_bytes):
    """Extracts all plain text from uploaded PDF bytes."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text.strip(), len(doc)


def merge_pdfs(pdf_list):
    """Merges a list of PDF bytes into a single PDF byte stream."""
    merged_doc = fitz.open()
    for pdf_bytes in pdf_list:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        merged_doc.insert_pdf(doc)
    output_stream = io.BytesIO()
    merged_doc.save(output_stream)
    return output_stream.getvalue()


def split_pdf(pdf_bytes, start_page, end_page):
    """Extracts a range of pages (1-indexed inclusive) into a new PDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    new_doc = fitz.open()
    # 0-indexed adjustment
    new_doc.insert_pdf(doc, from_page=start_page - 1, to_page=end_page - 1)
    output_stream = io.BytesIO()
    new_doc.save(output_stream)
    return output_stream.getvalue()

# ==============================================================================
# HELPER FUNCTIONS: ML & NLP ENGINES
# ==============================================================================

def extract_keywords_tfidf(text, top_n=10):
    """Extracts the most significant keywords using TF-IDF."""
    if len(text.split()) < 5:
        return []
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = np.array(vectorizer.get_feature_names_out())
    scores = tfidf_matrix.toarray()[0]
    top_indices = scores.argsort()[-top_n:][::-1]
    return [(feature_names[i], round(scores[i], 3)) for i in top_indices if scores[i] > 0]


def summarize_text_extractive(text, num_sentences=4):
    """Extractive summarization scoring sentences by TF-IDF token weights."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    clean_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if len(clean_sentences) <= num_sentences:
        return " ".join(clean_sentences)

    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(clean_sentences)
    except ValueError:
        return "Text too short or repetitive to summarize."

    # Score each sentence by summing its TF-IDF values
    sentence_scores = tfidf_matrix.sum(axis=1).A1
    top_indices = sorted(sentence_scores.argsort()[-num_sentences:])
    
    selected_sentences = [clean_sentences[i] for i in top_indices]
    return " ".join(selected_sentences)


def classify_document(text):
    """
    Classifies a document by calculating cosine similarity between
    the document text and predefined domain category keyword seeds.
    """
    categories = {
        "Technology & Computer Science": "programming software python algorithm database artificial intelligence machine learning network hardware data web cloud",
        "Business, Finance & Economics": "market revenue sales investment stock growth finance accounting profit banking quarterly shares economy management",
        "Legal & Regulatory": "law contract agreement jurisdiction court terms clause party statutory compliant regulation compliance liability",
        "Academic & Research": "methodology experiment analysis hypothesis citation results literature study paper journal evaluation participants",
        "Medical & Healthcare": "patient diagnosis clinical treatment disease medical surgery symptoms therapy health pharmaceutical hospital"
    }

    category_names = list(categories.keys())
    reference_texts = list(categories.values())
    all_docs = [text] + reference_texts

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    
    # Compare document vector (index 0) with category vectors (indices 1 to N)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    best_idx = int(np.argmax(similarities))
    confidence = similarities[best_idx] * 100

    results = []
    for i, name in enumerate(category_names):
        results.append((name, round(similarities[i] * 100, 2)))
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results[0][0], confidence, results


def compute_pdf_similarity(text1, text2):
    """Calculates cosine similarity percentage between two texts."""
    if not text1.strip() or not text2.strip():
        return 0.0
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([text1, text2])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(float(score) * 100, 2)

# ==============================================================================
# USER INTERFACE OLD (STREAMLIT)
# ==============================================================================

# Create a side-by-side layout for the logo and the title
col1, col2 = st.columns([1, 7], vertical_alignment="center")  # Adjust ratio as needed

with col1:
     st.image("Quire_dark.png", width=300)

with col2:
     st.title("QUIRE: PDF Editor and Analyzer")
     st.caption("BCA Mini Project | Sai Nath University | By Satyam")

st.markdown("---")  # Optional separator line below the header
tabs = st.tabs(["Smart ML Analysis", "PDF Similarity", "Convert to PDF", "PDF Merge & Split", "Text Extractor"])




# ------------------------------------------------------------------------------
# TAB 1: SMART ML ANALYSIS
# ------------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Document Summarization, Keywords & Classification")
    uploaded_file = st.file_uploader("Upload a PDF to analyze", type=["pdf"], key="ml_pdf")

    if uploaded_file:
        pdf_bytes = uploaded_file.read()
        extracted_text, num_pages = extract_text_from_pdf(pdf_bytes)

        st.info(f"Loaded **{uploaded_file.name}** ({num_pages} pages, {len(extracted_text.split())} words)")

        if st.button("Run ML Pipeline", type="primary"):
            if len(extracted_text.split()) < 30:
                st.warning("Document contains too few words for meaningful NLP analysis.")
            else:
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown("### Document Classification")
                    best_cat, confidence, all_scores = classify_document(extracted_text)
                    st.success(f"**Predicted Category:** {best_cat}")
                    st.progress(min(int(confidence * 1.5), 100))
                    
                    with st.expander("Show detailed category scores"):
                        for cat, score in all_scores:
                            st.write(f"- **{cat}**: {score}%")

                    st.markdown("### Key Topics & Keywords")
                    keywords = extract_keywords_tfidf(extracted_text, top_n=8)
                    for kw, score in keywords:
                        st.markdown(f"`{kw}` (score: {score})")

                with col2:
                    st.markdown("### Executive Summary (Extractive ML)")
                    summary = summarize_text_extractive(extracted_text, num_sentences=4)
                    st.write(summary)

# ------------------------------------------------------------------------------
# TAB 2: PDF SIMILARITY (PLAGIARISM / RELEVANCE CHECK)
# ------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Document Similarity & Overlap Detector")
    st.write("Compare two PDF documents to determine semantic similarity.")

    col_a, col_b = st.columns(2)
    with col_a:
        pdf_1 = st.file_uploader("Upload Base PDF (Doc A)", type=["pdf"], key="sim_1")
    with col_b:
        pdf_2 = st.file_uploader("Upload Comparison PDF (Doc B)", type=["pdf"], key="sim_2")

    if pdf_1 and pdf_2:
        if st.button("Compute Cosine Similarity"):
            t1, _ = extract_text_from_pdf(pdf_1.read())
            t2, _ = extract_text_from_pdf(pdf_2.read())
            sim_score = compute_pdf_similarity(t1, t2)

            st.metric(label="Calculated Similarity", value=f"{sim_score}%")
            if sim_score > 70:
                st.error("High similarity detected: Potential duplicate or heavily paraphrased content.")
            elif sim_score > 35:
                st.warning("Moderate similarity: Documents share overlapping subject matter.")
            else:
                st.success("Low similarity: Documents discuss distinct topics.")

# [Tabs 0, 1 remain unchanged...]

# ------------------------------------------------------------------------------
# TAB 3: CONVERT TO PDF (IMAGES & DOCS)
# ------------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Document & Image to PDF Converter")
    
    conv_subtab1, conv_subtab2 = st.tabs(["Images to PDF", "Word (DOCX) to PDF"])
    
    with conv_subtab1:
        st.write("Upload multiple images (PNG, JPG, JPEG) to combine into a single PDF.")
        uploaded_images = st.file_uploader("Choose images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        
        if uploaded_images:
            if st.button("Convert Images to PDF"):
                pdf_bytes = convert_images_to_pdf(uploaded_images)
                st.success("Images successfully converted to PDF!")
                st.download_button(
                    "Download Converted PDF", 
                    data=pdf_bytes, 
                    file_name="converted_images.pdf", 
                    mime="application/pdf"
                )

    with conv_subtab2:
        st.write("Upload a Microsoft Word document (.docx) to convert it into a PDF.")
        uploaded_docx = st.file_uploader("Choose Word document", type=["docx"])
        
        if uploaded_docx:
            if st.button("Convert DOCX to PDF"):
                pdf_bytes = convert_docx_to_pdf_bytes(uploaded_docx)
                st.success("Word document successfully converted to PDF!")
                st.download_button(
                    "Download Converted PDF", 
                    data=pdf_bytes, 
                    file_name="converted_document.pdf", 
                    mime="application/pdf"
                )

# [Keep the remaining Merge/Split and Text Extractor tabs unchanged, shifting their tab index accordingly if needed]

# ------------------------------------------------------------------------------
# TAB 4: PDF TOOLS (MERGE & SPLIT)
# ------------------------------------------------------------------------------
with tabs[3]:
    st.subheader("PDF Utilities")

    merge_subtab, split_subtab = st.tabs(["Merge PDFs", "Split / Slice PDF"])

    with merge_subtab:
        merge_files = st.file_uploader("Choose multiple PDFs to merge", type=["pdf"], accept_multiple_files=True, key="merge_files")
        if merge_files and len(merge_files) >= 2:
            if st.button("Merge Selected Files"):
                file_bytes_list = [f.read() for f in merge_files]
                merged_output = merge_pdfs(file_bytes_list)
                st.success("Merged successfully!")
                st.download_button("Download Merged PDF", data=merged_output, file_name="merged_output.pdf", mime="application/pdf")

    with split_subtab:
        split_file = st.file_uploader("Select a PDF to extract pages from", type=["pdf"], key="split_file")
        if split_file:
            split_bytes = split_file.read()
            doc = fitz.open(stream=split_bytes, filetype="pdf")
            total_pages = len(doc)
            st.write(f"Total Pages: **{total_pages}**")

            col_s, col_e = st.columns(2)
            with col_s:
                start_p = st.number_input("Start Page", min_value=1, max_value=total_pages, value=1)
            with col_e:
                end_p = st.number_input("End Page", min_value=1, max_value=total_pages, value=min(total_pages, 2))

            if st.button("Extract Pages"):
                if start_p > end_p:
                    st.error("Start page must be less than or equal to End page.")
                else:
                    split_output = split_pdf(split_bytes, start_p, end_p)
                    st.success(f"Pages {start_p} through {end_p} extracted!")
                    st.download_button("Download Extracted PDF", data=split_output, file_name=f"pages_{start_p}_to_{end_p}.pdf", mime="application/pdf")

# ------------------------------------------------------------------------------
# TAB 5: RAW TEXT EXTRACTOR
# ------------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Raw Text & Metadata Inspector")
    raw_pdf = st.file_uploader("Upload PDF to extract raw text", type=["pdf"], key="raw_file")
    if raw_pdf:
        text_out, count = extract_text_from_pdf(raw_pdf.read())
        st.write(f"Total Extracted Characters: **{len(text_out)}**")
        st.text_area("Extracted Plain Text", value=text_out, height=350)
        st.download_button("Download Text (.txt)", data=text_out, file_name="extracted_text.txt", mime="text/plain")

        


