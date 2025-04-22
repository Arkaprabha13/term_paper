import os
import pdfplumber
import nltk
import re
from nltk.tokenize import sent_tokenize
from groq import Groq
from info import GROQ_API_KEY, PDF_DIR, KEYWORDS

nltk.download('punkt')

client = Groq(api_key=GROQ_API_KEY)

PDF_DIR = PDF_DIR

SUMMARIES_DIR = os.path.join(PDF_DIR, "Summaries")
SHORT_SUMMARY_DIR = os.path.join(PDF_DIR, "Short_Summary")



# Chunk size to avoid API token limit
MAX_TOKENS = 3000  

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Function to extract URLs, data points, and references
def extract_links_and_data(text):
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

# Function to chunk text into smaller batches
def chunk_text(text, max_tokens=MAX_TOKENS):
    sentences = sent_tokenize(text)
    
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sent in sentences:
        sent_tokens = len(sent.split())
        
        if current_tokens + sent_tokens > max_tokens:
            chunks.append(current_chunk)
            current_chunk = sent
            current_tokens = sent_tokens
        else:
            current_chunk += " " + sent
            current_tokens += sent_tokens

    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

# Function to summarize text using Groq
def summarize_with_groq(text, context="general"):
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
        print(f"❌ Error: {e}")
        return "Failed to summarize due to API error."

# Function to generate short summary (200-300 words)
def generate_short_summary(text, max_length=300):
    sentences = sent_tokenize(text)
    
    ev_sentences = [sent for sent in sentences if any(kw in sent.lower() for kw in KEYWORDS)]
    
    short_summary = ""
    current_length = 0
    
    for sent in ev_sentences:
        words = len(sent.split())
        if current_length + words <= max_length:
            short_summary += sent + " "
            current_length += words
        else:
            break
    
    return short_summary.strip()

# ✅ Main execution
if __name__ == "__main__":
    # Create summary directories if they don't exist
    os.makedirs(SUMMARIES_DIR, exist_ok=True)
    os.makedirs(SHORT_SUMMARY_DIR, exist_ok=True)

    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".pdf"):
            print(f"📄 Processing: {filename}")
            
            pdf_path = os.path.join(PDF_DIR, filename)

            # Extract full text from PDF
            full_text = extract_text_from_pdf(pdf_path)

            # Extract links, data points, and references
            urls, data_points, references = extract_links_and_data(full_text)

            # Chunk the text for summarization
            chunks = chunk_text(full_text)

            # Initialize summaries
            full_summary = ""
            ev_disadvantages_summary = ""

            # Summarize each chunk
            for i, chunk in enumerate(chunks):
                print(f"📝 Summarizing chunk {i+1}/{len(chunks)}...")

                full_summary += f"\n### Chunk {i+1} Summary:\n"
                full_summary += summarize_with_groq(chunk, context="Full PDF Summary")

                ev_sentences = [sent for sent in sent_tokenize(chunk) if any(kw in sent.lower() for kw in KEYWORDS)]
                if ev_sentences:
                    ev_disadvantages_summary += f"\n### Chunk {i+1} EV Disadvantages:\n"
                    ev_disadvantages_summary += summarize_with_groq(" ".join(ev_sentences), context="EV Disadvantages Only")

            # ✅ Create individual folders
            base_filename = os.path.splitext(filename)[0]

            # Folder for detailed summary
            paper_folder = os.path.join(SUMMARIES_DIR, base_filename)
            os.makedirs(paper_folder, exist_ok=True)

            # Folder for short summary
            short_summary_folder = os.path.join(SHORT_SUMMARY_DIR, base_filename)
            os.makedirs(short_summary_folder, exist_ok=True)

            # ✅ Save detailed summaries
            txt_output_path = os.path.join(paper_folder, f"{base_filename}_summary.txt")
            md_output_path = os.path.join(paper_folder, f"{base_filename}_summary.md")

            # ✅ Save short summaries
            short_txt_output_path = os.path.join(short_summary_folder, f"{base_filename}_short_summary.txt")
            short_md_output_path = os.path.join(short_summary_folder, f"{base_filename}_short_summary.md")

            # Generate and save short summary
            short_summary = generate_short_summary(full_text)

            # Save detailed summaries
            with open(txt_output_path, "w", encoding="utf-8") as txt_file, \
                 open(md_output_path, "w", encoding="utf-8") as md_file:

                txt_file.write(f"\n📄 **File:** {filename}\n")
                txt_file.write("\n\n## 📝 Full Summary:\n")
                txt_file.write(full_summary)

                txt_file.write("\n\n## ⚠️ EV Disadvantages Summary:\n")
                txt_file.write(ev_disadvantages_summary)

                txt_file.write("\n\n## 🔗 Extracted Links:\n")
                txt_file.write("\n".join(urls) if urls else "No links found")

                txt_file.write("\n\n## 📊 Extracted Data Points:\n")
                txt_file.write("\n".join(data_points) if data_points else "No data points found")

                txt_file.write("\n\n## 📚 References/Proofs:\n")
                txt_file.write("\n".join([" - ".join(ref) for ref in references]) if references else "No references found")

            # Save short summaries
            with open(short_txt_output_path, "w", encoding="utf-8") as short_txt, \
                 open(short_md_output_path, "w", encoding="utf-8") as short_md:

                short_txt.write(f"\n📄 **File:** {filename}\n")
                short_txt.write("\n\n## 🔥 Short Summary:\n")
                short_txt.write(short_summary)

                short_md.write(f"# 📄 **File:** {filename}\n\n")
                short_md.write("## 🔥 **Short Summary**\n")
                short_md.write(short_summary)

            print(f"✅ Summarization completed for {filename}!")
            print(f"📁 Output saved in folder: {paper_folder} and {short_summary_folder}")

    print("\n🎉 All research papers have been processed successfully!")
