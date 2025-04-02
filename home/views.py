from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from decouple import config
import pdfplumber
import numpy as np
import pytesseract
import nltk
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from urllib.parse import urlparse, urljoin
import faiss
import os
import time

# Download stopwords (only once; cached on subsequent runs)
nltk.download("stopwords")
from nltk.corpus import stopwords

# ------------------ INITIALIZE FREE EMBEDDING MODEL ------------------ #
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dimension = 384  # Dimension for all-MiniLM-L6-v2

# ------------------ INITIALIZE FREE COMPLETION PIPELINE ------------------ #
from transformers import pipeline

# We'll use a text2text generation pipeline with FLAN-T5-base for generating answers.
qa_pipeline = pipeline(
    "text2text-generation", model="google/flan-t5-base", max_length=200, temperature=0.7
)


# ------------------ FAISS VECTOR DATABASE CLASS ------------------ #
class FAISSVectorDB:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = {}  # Maps vector id to metadata (text, document key, etc.)
        self.key_to_ids = {}  # Maps a unique document key to a list of vector ids

    def add_vectors(self, key, vectors, texts, extra_metadata=None):
        n = vectors.shape[0]
        if n == 0:
            return []
        start_id = self.index.ntotal
        self.index.add(vectors.astype(np.float32))
        vector_ids = list(range(start_id, start_id + n))
        for i, vec_id in enumerate(vector_ids):
            self.metadata[vec_id] = {
                "text": texts[i],
                "key": key,
                "extra": extra_metadata,
            }
        if key in self.key_to_ids:
            self.key_to_ids[key].extend(vector_ids)
        else:
            self.key_to_ids[key] = vector_ids
        return vector_ids

    def exists(self, key):
        return key in self.key_to_ids

    def get_chunks_by_key(self, key):
        if key not in self.key_to_ids:
            return []
        ids = self.key_to_ids[key]
        return [self.metadata[i]["text"] for i in ids]

    def search(self, query_vector, k=3):
        query_vector = np.array(query_vector, dtype=np.float32).reshape(
            1, self.dimension
        )
        distances, indices = self.index.search(query_vector, k)
        results = []
        for d, idx in zip(distances[0], indices[0]):
            if idx in self.metadata:
                results.append((d, self.metadata[idx]["text"]))
        return results


# Global FAISS vector database instance
VECTOR_DB = FAISSVectorDB(dimension=embedding_dimension)


# ------------------ HELPER FUNCTIONS ------------------ #
def normalize_url(url):
    parsed_url = urlparse(url)
    return f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"


def extract_text_from_pdf(file_obj):
    text = ""
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if not page_text or page_text.strip() == "":
                try:
                    page_image = page.to_image()
                    pil_image = page_image.original
                    ocr_text = pytesseract.image_to_string(pil_image)
                    text += ocr_text + "\n"
                except Exception as e:
                    text += ""
            else:
                text += page_text + "\n"
    return text


def extract_text_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if "application/pdf" in content_type:
            file_bytes = BytesIO(response.content)
            return extract_text_from_pdf(file_bytes)
        elif "text/html" in content_type:
            soup = BeautifulSoup(response.text, "html.parser")
            return soup.get_text(separator="\n")
        else:
            return ""
    except Exception as e:
        return ""


def create_chunks(text, chunk_size=50, overlap=10):
    """Split the text into smaller chunks with chunk_size words and an overlap of overlap words."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start : start + chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    words = text.split()
    filtered_words = [w for w in words if w.lower() not in stop_words]
    return " ".join(filtered_words)


def get_embeddings(texts):
    """
    Batch request embeddings for a list of texts using SentenceTransformers.
    Returns a list of embedding vectors.
    """
    try:
        # encoding returns a numpy array of shape (n_texts, dimension)
        embeddings = embedding_model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()  # convert to list of lists
    except Exception as e:
        print(f"Error in batch embedding: {e}")
        return [None] * len(texts)


def get_embedding(text):
    """Get an embedding for a single text using batch embedding."""
    embeddings = get_embeddings([text])
    if embeddings and embeddings[0] is not None:
        return embeddings[0]
    return None


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)


def get_vector_db_key_for_pdf(file_obj):
    return "pdf:" + file_obj.name


def get_vector_db_key_for_url(url):
    norm_url = normalize_url(url)
    return "url:" + norm_url


def process_pdf(file_obj):
    key = get_vector_db_key_for_pdf(file_obj)
    if VECTOR_DB.exists(key):
        return VECTOR_DB.get_chunks_by_key(key)
    text = extract_text_from_pdf(file_obj)
    chunks = create_chunks(text, chunk_size=50, overlap=10)
    cleaned_chunks = [remove_stop_words(chunk) for chunk in chunks if chunk.strip()]
    embeddings = get_embeddings(cleaned_chunks)
    valid_embeddings = []
    valid_texts = []
    for chunk, emb in zip(cleaned_chunks, embeddings):
        if emb is None:
            print("Skipping chunk due to batch embedding error")
            continue
        if len(emb) != VECTOR_DB.dimension:
            print(
                f"Skipping chunk: got dimension {len(emb)}, expected {VECTOR_DB.dimension}"
            )
            continue
        valid_embeddings.append(emb)
        valid_texts.append(chunk)
    if len(valid_embeddings) == 0:
        return []
    vectors = np.array(valid_embeddings)
    VECTOR_DB.add_vectors(
        key, vectors, valid_texts, extra_metadata={"type": "pdf", "name": file_obj.name}
    )
    return valid_texts


def process_url(url, max_chunks=20):
    key = get_vector_db_key_for_url(url)
    if VECTOR_DB.exists(key):
        return VECTOR_DB.get_chunks_by_key(key)
    text = extract_text_from_url(url)
    chunks = create_chunks(text, chunk_size=50, overlap=10)
    # Limit the number of chunks per URL
    chunks = chunks[:max_chunks]
    cleaned_chunks = [remove_stop_words(chunk) for chunk in chunks if chunk.strip()]
    embeddings = get_embeddings(cleaned_chunks)
    valid_embeddings = []
    valid_texts = []
    for chunk, emb in zip(cleaned_chunks, embeddings):
        if emb is None:
            print("Skipping chunk from URL due to batch embedding error")
            continue
        if len(emb) != VECTOR_DB.dimension:
            print(
                f"Skipping chunk from URL: got dimension {len(emb)}, expected {VECTOR_DB.dimension}"
            )
            continue
        valid_embeddings.append(emb)
        valid_texts.append(chunk)
    if len(valid_embeddings) == 0:
        return []
    vectors = np.array(valid_embeddings)
    VECTOR_DB.add_vectors(
        key,
        vectors,
        valid_texts,
        extra_metadata={"type": "url", "url": normalize_url(url)},
    )
    return valid_texts


# ------------------ DJANGO VIEW ------------------ #
@csrf_exempt
def chat_view(request):
    if request.method == "GET":
        return render(request, "index.html")
    elif request.method == "POST":
        question = request.POST.get("question")
        if not question:
            return JsonResponse({"error": "No question provided"}, status=400)
        all_chunks = []
        # Process uploaded PDFs
        for key, file_obj in request.FILES.items():
            if file_obj.name.lower().endswith(".pdf"):
                chunks = process_pdf(file_obj)
                all_chunks.extend(chunks)
        # Process URL inputs (all POST keys starting with "url_")
        for key, value in request.POST.items():
            if key.startswith("url_"):
                url = value.strip()
                if url:
                    chunks = process_url(url)
                    all_chunks.extend(chunks)
        if not all_chunks:
            return JsonResponse(
                {"error": "No valid PDF or URL content found."}, status=400
            )
        cleaned_question = remove_stop_words(question)
        question_embedding = get_embedding(cleaned_question)
        if question_embedding is None:
            return JsonResponse({"error": "Failed to embed question."}, status=400)
        results = VECTOR_DB.search(question_embedding, k=3)
        top_chunks = [text for _, text in results]
        context = "\n".join(top_chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        try:
            # Use the free generation pipeline to generate an answer.
            # We use the qa_pipeline from transformers.
            completion = qa_pipeline(prompt)
            answer = completion[0]["generated_text"].strip()
        except Exception as e:
            answer = f"Error during generation: {str(e)}"
        return JsonResponse({"answer": answer})
    else:
        return JsonResponse({"error": "Invalid request method"}, status=400)
