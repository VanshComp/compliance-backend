import os
import base64
import datetime
import uuid
import time
import json
import re
import logging
from typing import List, Dict, Any

from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
from docx import Document
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil

load_dotenv()

import os

LOGO_GUIDELINES_PDF_PATH = os.path.join(os.path.dirname(__file__), "JSW Brand Guidelines.pdf")
RULES_PDF_PATH = os.path.join(os.path.dirname(__file__), "COMP64980.pdf")

from mangum import Mangum

handler = Mangum(app)

# Helper functions defined first
def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in ['.txt', '.md', '.csv']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == '.pdf':
            reader = PdfReader(file_path)
            text = '\n\n'.join(page.extract_text() or '' for page in reader.pages)
            if not text.strip():
                logging.warning("Scanned PDF detected; limited text extraction.")
            return text
        elif ext in ['.jpg', '.png', '.jpeg']:
            return pytesseract.image_to_string(Image.open(file_path))
        elif ext == '.docx':
            doc = Document(file_path)
            return '\n\n'.join(para.text for para in doc.paragraphs)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        logging.error(f"Error extracting text from {file_path}: {str(e)}")
        raise

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    words = re.findall(r'\S+', text)
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def get_embedding(text: str) -> List[float]:
    try:
        response = client.embeddings.create(input=text, model='text-embedding-3-small')
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Embedding error: {str(e)}")
        raise

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Unified Pinecone config
INDEX_NAME = "jsw-brand-compliance"
NAMESPACE_LOGO_REPO = "logo-repository"
NAMESPACE_LOGO_GUIDELINES = "logo-guidelines"
NAMESPACE_TEXT_RULES = "obpp-compliance-rules"
DIMENSION = 1536
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create or validate the single index
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    index_desc = pc.describe_index(INDEX_NAME)
    if index_desc.dimension != DIMENSION:
        pc.delete_index(INDEX_NAME)
        while INDEX_NAME in pc.list_indexes().names():
            time.sleep(1)
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

index = pc.Index(INDEX_NAME)

# Paths
LOGO_GUIDELINES_PDF_PATH = "C:\\Users\\User\\OneDrive\\Desktop\\AryanYadav\\jswcentura_backend\\JSW Brand Guidelines.pdf"
RULES_PDF_PATH = "C:\\Users\\User\\OneDrive\\Desktop\\AryanYadav\\jswcentura_backend\\COMP64980.pdf"

# Store guidelines if namespace is empty
stats = index.describe_index_stats()
if stats.namespaces.get(NAMESPACE_LOGO_GUIDELINES, {'vector_count': 0})['vector_count'] == 0:
    print("Storing logo guidelines in Pinecone...")
    def store_guidelines(guidelines_pdf_path: str) -> None:
        guidelines_text = extract_text_from_file(guidelines_pdf_path)
        guidelines_text = re.sub(r'\s+', ' ', guidelines_text).strip()
        chunks = chunk_text(guidelines_text)
        vectors = []
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            vectors.append({
                'id': f'guideline_{i}',
                'values': embedding,
                'metadata': {'text': chunk}
            })
        if vectors:
            index.upsert(vectors=vectors, namespace=NAMESPACE_LOGO_GUIDELINES)
            logging.info(f"Stored {len(chunks)} guideline chunks in Pinecone namespace {NAMESPACE_LOGO_GUIDELINES}.")
        else:
            logging.warning("No chunks to store.")
    store_guidelines(LOGO_GUIDELINES_PDF_PATH)

# Store text rules if namespace is empty
if stats.namespaces.get(NAMESPACE_TEXT_RULES, {'vector_count': 0})['vector_count'] == 0:
    print("Storing text rules in Pinecone...")
    def store_rules(rules_pdf_path: str) -> None:
        rules_text = extract_text_from_file(rules_pdf_path)
        rules_text = re.sub(r'\s+', ' ', rules_text).strip()
        chunks = chunk_text(rules_text)
        vectors = []
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            vectors.append({
                'id': f'rule_{i}',
                'values': embedding,
                'metadata': {'text': chunk}
            })
        if vectors:
            index.upsert(vectors=vectors, namespace=NAMESPACE_TEXT_RULES)
            logging.info(f"Stored {len(chunks)} rule chunks in Pinecone namespace {NAMESPACE_TEXT_RULES}.")
        else:
            logging.warning("No chunks to store.")
    store_rules(RULES_PDF_PATH)

CRITERIA = """
- Logo Analysis:
  - Vector Dimension Verification
  - Color Compliance Checking
  - Pixel-Level Quality Assessment
  - Position and Alignment Validation

- Font Verification:
  - OTF (Open Type Font) Format Support
  - Integration with Company Font Library
  - Size and Style Compliance

- Color Compliance:
  - HEX Code Validation
  - Saturation and Highlight Checks
  - Gradient and Texture Alignment

- Gradients/Textures:
  - Background or Subject Gradients
  - Spacing and Style Matching

- Content Repository Management:
  - Automated Updates
  - Brand Asset Version Control
  - Regular Verification Cycles
"""

def describe_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that describes images in detail."},
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this image in detail, focusing on elements relevant to a brand logo such as shapes, colors, fonts, and overall design."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        max_tokens=300
    )
    return response.choices[0].message.content

def embed_image(image_path):
    description = describe_image(image_path)
    response = client.embeddings.create(
        input=description,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    return embedding

def store_in_pinecone_logo(embedding, metadata):
    vector_id = str(uuid.uuid4())
    index.upsert(vectors=[(vector_id, embedding, metadata)], namespace=NAMESPACE_LOGO_REPO)
    print(f"Logo stored in Pinecone with ID: {vector_id}")
    return vector_id

def analyze_logo_with_llm(image_path):
    description = describe_image(image_path)
    embedding = get_embedding(description)
    query_response = index.query(vector=embedding, top_k=10, include_metadata=True, namespace=NAMESPACE_LOGO_GUIDELINES)
    relevant_guidelines = '\n\n'.join(match['metadata'].get('text', '') for match in query_response['matches'] if 'metadata' in match and match['score'] > 0.7)
    
    prompt = f"""
You are an expert in brand logo analysis, specializing in JSW brand guidelines.
Here is a description of the uploaded logo: "{description}"

Relevant JSW Brand Guidelines: "{relevant_guidelines}"

Evaluate the logo against the following criteria, incorporating the JSW-specific guidelines where applicable:
{CRITERIA}

For each main criterion category (Logo Analysis, Font Verification, Color Compliance, Gradients/Textures, Content Repository Management), provide:
- A sub-list of evaluations for each sub-criterion, including:
  - Pass/Fail
  - A score from 0 to 10
- Then, calculate category_percentage as the average of sub-criterion scores multiplied by 10.
- Assign status: "Pass" if category_percentage >= 80, "Warning" if 50 <= category_percentage < 80, "Fail" if category_percentage < 50.
- Flag any obvious mismatches or anomalies (e.g., incorrect colors, fonts, or design elements that deviate from JSW brand standards) that indicate the logo might be wrong or misaligned.

Assume the logo is intended to represent the JSW brand and penalize heavily (score 0-3) for any significant deviations such as mismatched colors, inappropriate fonts, or poor design quality based on the guidelines. If the logo appears non-compliant with JSW standards (e.g., wrong positioning, tagline usage, colors), adjust scores downward accordingly.

Then:
- Calculate overall accuracy percentage as the average of all category_percentages.
- Provide overall improvements or suggestions, with specific recommendations if the logo is deemed incorrect or substandard, referencing JSW guidelines.
- Provide what is right: thorough and detailed text explaining all the positive aspects of the logo that are compliant, how they contribute to the category percentages, and reference specific sections or rules from the uploaded JSW guidelines. Be exhaustive in referencing the guidelines to show compliance.

Output the response strictly in JSON format with the following structure:
{{
  "evaluations": [
    {{
      "category": "Logo Analysis",
      "sub_criteria": [
        {{
          "name": "Vector Dimension Verification",
          "pass_fail": "Pass" or "Fail",
          "score": number
        }},
        ... (for each sub-criterion)
      ],
      "category_percentage": number,
      "status": "Pass" or "Warning" or "Fail"
    }},
    ... (for each category)
  ],
  "overall_accuracy_percentage": number,
  "improvements": "text suggestions here",
  "anomalies_detected": ["list of anomalies or empty list if none"],
  "what_is_right": "detailed explanation text here"
}}
Do not include any additional text outside the JSON.
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in brand logo analysis."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    analysis_json = json.loads(response.choices[0].message.content)
    return analysis_json

def process_logo(image_path, brand_name="JSW"):
    embedding = embed_image(image_path)
    metadata = {
        "brand": brand_name,
        "timestamp": str(datetime.datetime.now()),
        "file_path": image_path
    }
    vector_id = store_in_pinecone_logo(embedding, metadata)
    analysis = analyze_logo_with_llm(image_path)
    percentage = analysis.get("overall_accuracy_percentage", 0.0)
    anomalies = analysis.get("anomalies_detected", [])
    metadata["analysis"] = json.dumps(analysis)
    metadata["approval_percentage"] = percentage
    index.update(id=vector_id, set_metadata=metadata, namespace=NAMESPACE_LOGO_REPO)
    return analysis

# -------------------- Text Compliance Functions --------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_compliance(input_files: List[str], chunk_size: int = 500) -> List[Dict[str, Any]]:
    overall_results = []
    for file_path in input_files:
        try:
            input_text = extract_text_from_file(file_path)
            input_text = re.sub(r'\s+', ' ', input_text).strip()
            chunks = chunk_text(input_text, chunk_size)
            if not chunks:
                overall_results.append({'file': file_path, 'overall_accuracy': 100.0, 'overall_improvements': ['No content to check.'], 'overall_what_is_right': 'No content to check.', 'overall_anomalies_detected': []})
                continue

            file_results: Dict[str, Any] = {'file': file_path, 'chunks': []}
            total_score = 0.0
            num_chunks = len(chunks)
            all_what_is_right = []
            all_anomalies = set()
            all_improvements = set()
            for chunk_idx, chunk in enumerate(chunks):
                try:
                    embedding = get_embedding(chunk)
                    query_response = index.query(vector=embedding, top_k=5, include_metadata=True, namespace=NAMESPACE_TEXT_RULES)
                    relevant_rules = '\n\n'.join(match['metadata'].get('text', '') for match in query_response['matches'] if 'metadata' in match and match['score'] > 0.8)
                    prompt = f"""
                    Rules: {relevant_rules}

                    Input Text: {chunk}

                    Task: Objectively check the Input Text against the Rules for compliance with OBPP Advertisement Code.
                    - Analyze for violations (e.g., missing disclosures, prohibited terms), but only flag clear non-compliance.
                    - If the text is not an advertisement, note that and give 100% if no issues.
                    - Compute accuracy rate: integer (0-100) of how compliant the text is (100 = no violations).
                    - List improvements: array of specific changes to make it 100% compliant. If none, empty array.
                    - List anomalies: array of specific violations or anomalies detected (e.g., prohibited terms, missing disclosures). If none, empty array.
                    - Provide what_is_right: thorough and detailed text explaining all the positive aspects of the text that are compliant, referencing specific sections or rules from the uploaded guidelines where applicable. Be exhaustive in referencing the guidelines to show compliance. If none, empty string.

                    Output ONLY JSON:
                    {{
                      "accuracy": <int>,
                      "improvements": [<str>, ...],
                      "anomalies": [<str>, ...],
                      "what_is_right": "<str>"
                    }}
                    """
                    response = client.chat.completions.create(
                        model='gpt-4o',
                        messages=[{'role': 'user', 'content': prompt}],
                        response_format={"type": "json_object"}
                    )
                    output = response.choices[0].message.content
                    parsed = json.loads(output)
                    accuracy = int(parsed.get('accuracy', 0))
                    improvements = parsed.get('improvements', [])
                    anomalies = parsed.get('anomalies', [])
                    what_is_right = parsed.get('what_is_right', '')
                    file_results['chunks'].append({
                        'chunk_idx': chunk_idx,
                        'chunk': chunk,
                        'accuracy': accuracy,
                        'improvements': improvements,
                        'anomalies': anomalies,
                        'what_is_right': what_is_right
                    })
                    total_score += accuracy
                    if what_is_right:
                        all_what_is_right.append(what_is_right)
                    all_improvements.update(improvements)
                    all_anomalies.update(anomalies)
                except Exception as e:
                    logging.error(f"Error processing chunk {chunk_idx} of {file_path}: {str(e)}")
                    file_results['chunks'].append({
                        'chunk_idx': chunk_idx,
                        'chunk': chunk,
                        'accuracy': 0,
                        'improvements': [f"Error: {str(e)}"],
                        'anomalies': [],
                        'what_is_right': ''
                    })
                    total_score += 0
            avg_accuracy = total_score / num_chunks if num_chunks > 0 else 0.0
            file_results['overall_accuracy'] = avg_accuracy
            file_results['overall_improvements'] = sorted(list(all_improvements))
            file_results['overall_anomalies_detected'] = sorted(list(all_anomalies))
            file_results['overall_what_is_right'] = '\n\n'.join(all_what_is_right)
            overall_results.append(file_results)
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            overall_results.append({'file': file_path, 'overall_accuracy': 0.0, 'overall_improvements': [f"File error: {str(e)}"], 'overall_anomalies_detected': [], 'overall_what_is_right': ''})
    return overall_results

# -------------------- API Endpoints --------------------

@app.post("/check-logo")
async def check_logo_endpoint(file: UploadFile = File(...)):
    logging.info(f"Received file: {file.filename}, size: {file.size} bytes")
    temp_path = f"temp_{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        analysis = process_logo(temp_path)
        logging.info(f"Logo analysis completed for {file.filename}: {json.dumps(analysis)}")
        return analysis
    except Exception as e:
        logging.error(f"Error processing logo: {str(e)}")
        raise
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/check-text")
async def check_text_endpoint(file: UploadFile = File(None), text: str = Form(None)):
    logging.info(f"Received text file: {file.filename if file else 'none'}, text length: {len(text or '')}")
    temp_path = None
    try:
        if file:
            temp_path = f"temp_{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        elif text:
            temp_path = f"temp_{uuid.uuid4()}.txt"
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(text)
        
        results = check_compliance([temp_path]) if temp_path else [{'overall_accuracy': 100.0, 'overall_improvements': ['No content to check.'], 'overall_anomalies_detected': [], 'overall_what_is_right': 'No content to check.'}]
        logging.info(f"Text analysis completed: {json.dumps(results[0])}")
        return results[0]
    except Exception as e:
        logging.error(f"Error processing text: {str(e)}")
        raise
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)