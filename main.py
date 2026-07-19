import os
import time
import logging
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

# ---------------------------------------------------------
# Setup Professional Logging
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables (API Keys)
load_dotenv()


class MicroRAG:
    """
    A minimalist, from-scratch Retrieval-Augmented Generation (RAG) pipeline.
    Built to demonstrate core AI infrastructure mechanics including text chunking,
    vector embeddings, semantic search, and prompt augmentation.
    """

    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2', llm_model: str = 'gemini-2.0-flash'):
        self.llm_model = llm_model
        
        # Initialize Google GenAI Client
        if not os.environ.get("GEMINI_API_KEY"):
            logger.warning("GEMINI_API_KEY not found in environment variables!")
        self.client = genai.Client()
        
        # Initialize local embedding model
        logger.info(f"Loading embedding model '{embedding_model_name}'...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        logger.info("System Initialization Complete.\n")

    # --- PHASE 1: INGESTION ---
    def chunk_document(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Splits a long document into smaller, overlapping semantic chunks."""
        chunks = []
        start_index = 0
        text_length = len(text)

        while start_index < text_length:
            end_index = start_index + chunk_size
            chunk = text[start_index:end_index]
            chunks.append(chunk)
            start_index = end_index - overlap

        return chunks

    # --- PHASE 2: EMBEDDING ---
    def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        """Converts raw text chunks into dense mathematical vectors."""
        logger.info(f"Translating {len(chunks)} text chunks into high-dimensional vectors...")
        raw_embeddings = self.embedding_model.encode(chunks)
        return raw_embeddings.tolist()

    # --- PHASE 3: VECTOR SEARCH ---
    @staticmethod
    def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Measures the semantic angle (similarity) between two vectors."""
        v1, v2 = np.array(vec1), np.array(vec2)
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return dot_product / (norm_v1 * norm_v2)

    def retrieve_top_chunks(self, query: str, chunks: List[str], chunk_vectors: List[List[float]], top_k: int = 2) -> List[Dict[str, Any]]:
        """Finds the most relevant context chunks for a given user query."""
        logger.info(f"Embedding user query: '{query}'")
        query_vector = self.embedding_model.encode(query).tolist()

        scored_results = []
        for index, chunk_vec in enumerate(chunk_vectors):
            score = self.calculate_cosine_similarity(query_vector, chunk_vec)
            scored_results.append({
                "score": score,
                "text": chunks[index]
            }) 

        # Sort by highest similarity score
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]

    # --- PHASE 4: GENERATION ---
    def generate_answer(self, query: str, retrieved_results: List[Dict[str, Any]], max_retries: int = 3) -> str:
        """Assembles the retrieved facts into a prompt and calls the LLM with retry logic."""
        context_text = "\n".join([f"- {match['text']}" for match in retrieved_results])
        
        augmented_prompt = f"""System: You are a strict corporate knowledge assistant. You must answer the user's question using ONLY the facts provided in the Context section below. If the context does not contain the answer, reply exactly with: "I am sorry, but I do not have access to that information in the current database." Do not make anything up.

Context:
{context_text}

User Question: {query}
Answer:"""
        
        logger.info("Transmitting augmented payload to Gemini API...")
        
        # Exponential Backoff Retry Logic
        base_delay = 5
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.llm_model,
                    contents=augmented_prompt
                )
                return response.text
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    if attempt == max_retries - 1:
                        return f"API Error: Rate limit exhausted."
                    wait_time = base_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    return f"API Error: {error_msg}"


# ---------------------------------------------------------
# Execution Block
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Initializing End-to-End MicroRAG Pipeline")
    print("="*50 + "\n")

    # 1. Instantiate our pipeline
    rag_system = MicroRAG()

    # 2. Define the Knowledge Base (Corpus)
    knowledge_base = [
        "The company cafeteria serves pizza on Fridays.",
        "RAG systems combine retrieval and generation.",
        "Employees are required to use two-factor authentication.",
        "The CEO's name is Sarah Jenkins."
    ]

    # 3. Process the Data (Ingest & Embed)
    logger.info("Processing Knowledge Base...")
    kb_vectors = rag_system.embed_chunks(knowledge_base)

    # 4. User Interaction
    user_query = "Who is the chief executive officer?"
    
    # 5. Retrieve Context
    logger.info("Running Vector Search Engine...")
    top_matches = rag_system.retrieve_top_chunks(user_query, knowledge_base, kb_vectors, top_k=1)

    # 6. Generate Response
    final_response = rag_system.generate_answer(user_query, top_matches)

    # 7. Output
    print("\n" + "="*50)
    print("LLM FINAL RESPONSE:")
    print("="*50)
    print(final_response)
    print("\n")
