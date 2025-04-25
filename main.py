"""
Description:
This script automates the extraction, summarization, and structured reporting 
of research papers in PDF format. It uses Groq's LLaMA-3.3-70B model via API 
for generating both detailed summaries and focused summaries on predefined 
keywords such as EV (Electric Vehicle) disadvantages.

Features:
- Extracts full text from PDFs  
- Detects and collects links, data points, and reference mentions  
- Generates AI-powered summaries in both detailed and concise formats  
- Organizes output neatly in structured directories  

KEYWORDS = [
    # Environmental concerns
    "lithium mining", "cobalt mining", "nickel mining", "rare earth metals",
    "excessive mining", "toxic chemicals", "hazardous waste", "battery waste",
    "battery disposal", "battery leakage", "metal toxicity", "water pollution",
    "soil degradation", "air pollution", "noise pollution",

    # Energy and emissions
    "carbon footprint", "coal-based electricity", "emissions from power generation",
    "fossil fuel dependency", "non-renewable energy dependency", "charging station emissions",

    # Recycling and disposal issues
    "recycling inefficiency", "recycling challenges", "battery recalls",

    # Supply chain and ethical concerns
    "child labor", "ethical concerns", "displacement of local communities", "supply chain issues",

    # Electrical and grid issues
    "charging infrastructure", "grid strain", "grid instability", "electricity shortages",
    "power outages", "transformer aging", "local power cuts", "energy storage inefficiency",

    # Performance and reliability
    "limited range", "charging time delays", "limited charging cycles", "battery degradation",
    "battery failure", "thermal runaway", "overheating", "thermal issues", "fire hazard",
    "battery overheating incidents", "battery explosion", "driving risks", "accident severity",
    "cold weather inefficiency", "extreme heat issues", "durability issues", "battery lifespan concerns",
    "maintenance challenges", "high repair costs", "high maintenance costs",

    # Cost-related concerns
    "high costs", "insurance risks", "infrastructure challenges", "installation costs",
    "operational costs", "battery replacement cost",

    # Technology-related concerns
    "electromagnetic interference", "circuit damage", "charging station availability",

    # Hidden environmental burdens
    "emissions displacement", "mining ecosystem damage", "local water depletion", "air quality impact",

    # Power generation and grid-level challenges
    "load balancing problems", "power plant capacity stress", "transformer failure risks",
    "transmission losses", "emergency backup capacity issues", "grid load forecasting challenges"
]

Dependencies:
- pdfplumber  
- nltk  
- re  
- Groq Python SDK  

* Markdown files (.md) just for easy formatting for readability:
"""

import os
import pdfplumber
import nltk
import re
from nltk.tokenize import sent_tokenize
from groq import Groq
from info import GROQ_API_KEY, PDF_DIR, KEYWORDS

# Download necessary NLTK models
nltk.download('punkt')

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Define directories
SUMMARIES_DIR = os.path.join(PDF_DIR, "Summaries")
SHORT_SUMMARY_DIR = os.path.join(PDF_DIR, "Short_Summary")

# Chunk size to avoid API token limits
MAX_TOKENS = 3000  

def extract_text_from_pdf(pdf_path):
    """
    Extract text content from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    with pdfplumber.open(pdf_path) as pdf:
        text = "".join(page.extract_text() or "" for page in pdf.pages)
    return text


def extract_links_and_data(text):
    """
    Extract URLs, numeric data points, and reference mentions from text.

    Args:
        text (str): Input text.

    Returns:
        tuple: URLs (list), data points (list), and references (list).
    """
    urls = re.findall(r'(https?://[^\s]+)', text)

    data_points = []
    sentences = sent_tokenize(text)
    for sentence in sentences:
        numbers = re.findall(r'\b\d+(\.\d+)?\b', sentence)
        if numbers:
            for num in numbers:
                data_points.append(f"{num} -> {sentence.strip()}")

    references = re.findall(r'(Reference[s]?|Cite[d]?|Source[s]?)[:\s]+(.+)', text, re.IGNORECASE)

    return urls, data_points, references


def chunk_text(text, max_tokens=MAX_TOKENS):
    """
    Split text into smaller chunks based on sentence boundaries.

    Args:
        text (str): Input text.
        max_tokens (int): Maximum number of tokens per chunk.

    Returns:
        list: List of text chunks.
    """
    sentences = sent_tokenize(text)
    chunks, current_chunk, current_tokens = [], "", 0

    for sent in sentences:
        sent_tokens = len(sent.split())

        if current_tokens + sent_tokens > max_tokens:
            chunks.append(current_chunk)
            current_chunk, current_tokens = sent, sent_tokens
        else:
            current_chunk += " " + sent
            current_tokens += sent_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def summarize_with_groq(text, context="general"):
    """
    Generate a summary using Groq's LLaMA-3.3-70B model.

    Args:
        text (str): Input text.
        context (str): Summary context.

    Returns:
        str: Generated summary.
    """
    prompt = (
        f"Summarize the following research paper content. Context: {context}. "
        f"Use only the information from the text and avoid generalizations. "
        f"Include any links, data, or proofs mentioned in the content.\n\n"
        f"Text:\n{text}\n\n"
    )

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-specdec",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4096
        )
        return completion.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error: API Summarization failed. Details: {e}")
        return "Failed to summarize due to API error."


def generate_short_summary(text, max_length=300):
    """
    Generate a short summary focusing on sentences containing specific keywords.

    Args:
        text (str): Input text.
        max_length (int): Maximum word length of the summary.

    Returns:
        str: Generated short summary.
    """
    sentences = sent_tokenize(text)
    ev_sentences = [sent for sent in sentences if any(kw in sent.lower() for kw in KEYWORDS)]

    short_summary, current_length = "", 0
    for sent in ev_sentences:
        words = len(sent.split())
        if current_length + words <= max_length:
            short_summary += sent + " "
            current_length += words
        else:
            break

    return short_summary.strip()


if __name__ == "__main__":
    print("\nResearch Paper Summarization Process Started.\n")

    # Create output directories
    os.makedirs(SUMMARIES_DIR, exist_ok=True)
    os.makedirs(SHORT_SUMMARY_DIR, exist_ok=True)

    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".pdf"):
            print(f"Processing file: {filename}")

            pdf_path = os.path.join(PDF_DIR, filename)

            full_text = extract_text_from_pdf(pdf_path)
            urls, data_points, references = extract_links_and_data(full_text)
            chunks = chunk_text(full_text)

            full_summary, ev_disadvantages_summary = "", ""

            for i, chunk in enumerate(chunks):
                print(f"Summarizing chunk {i+1} of {len(chunks)}.")

                full_summary += f"\n### Chunk {i+1} Summary:\n"
                full_summary += summarize_with_groq(chunk, context="Full PDF Summary")

                ev_sentences = [sent for sent in sent_tokenize(chunk)
                                if any(kw in sent.lower() for kw in KEYWORDS)]
                if ev_sentences:
                    ev_disadvantages_summary += f"\n### Chunk {i+1} EV Disadvantages:\n"
                    ev_disadvantages_summary += summarize_with_groq(" ".join(ev_sentences),
                                                                   context="EV Disadvantages Only")

            base_filename = os.path.splitext(filename)[0]
            paper_folder = os.path.join(SUMMARIES_DIR, base_filename)
            short_summary_folder = os.path.join(SHORT_SUMMARY_DIR, base_filename)
            os.makedirs(paper_folder, exist_ok=True)
            os.makedirs(short_summary_folder, exist_ok=True)

            # Detailed summary files
            txt_output_path = os.path.join(paper_folder, f"{base_filename}_summary.txt")
            md_output_path = os.path.join(paper_folder, f"{base_filename}_summary.md")

            # Short summary files
            short_txt_output_path = os.path.join(short_summary_folder, f"{base_filename}_short_summary.txt")
            short_md_output_path = os.path.join(short_summary_folder, f"{base_filename}_short_summary.md")

            short_summary = generate_short_summary(full_text)

            with open(txt_output_path, "w", encoding="utf-8") as txt_file, \
                 open(md_output_path, "w", encoding="utf-8") as md_file:

                txt_file.write(f"\nFile: {filename}\n")
                txt_file.write("\n\n## Full Summary:\n")
                txt_file.write(full_summary)

                txt_file.write("\n\n## EV Disadvantages Summary:\n")
                txt_file.write(ev_disadvantages_summary)

                txt_file.write("\n\n## Extracted Links:\n")
                txt_file.write("\n".join(urls) if urls else "No links found")

                txt_file.write("\n\n## Extracted Data Points:\n")
                txt_file.write("\n".join(data_points) if data_points else "No data points found")

                txt_file.write("\n\n## References/Proofs:\n")
                txt_file.write("\n".join([" - ".join(ref) for ref in references]) if references else "No references found")

            with open(short_txt_output_path, "w", encoding="utf-8") as short_txt, \
                 open(short_md_output_path, "w", encoding="utf-8") as short_md:

                short_txt.write(f"\nFile: {filename}\n")
                short_txt.write("\n\n## Short Summary:\n")
                short_txt.write(short_summary)

                short_md.write(f"# File: {filename}\n\n")
                short_md.write("## Short Summary\n")
                short_md.write(short_summary)

            print(f"Summarization completed for: {filename}")
            print(f"Output saved in:\n   - {paper_folder}\n   - {short_summary_folder}\n")

    print("All research papers have been processed successfully.\n")
