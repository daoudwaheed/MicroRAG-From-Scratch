# MicroRAG: From-Scratch Retrieval-Augmented Generation

A minimalist, dependency-light Retrieval-Augmented Generation (RAG) pipeline built in pure Python. 

This project was built to demonstrate the core architectural mechanics of enterprise AI systems. Instead of abstracting the underlying logic behind heavy frameworks like LangChain or LlamaIndex, this repository explicitly codes the math, data flows, and API handling from scratch.

## Architecture & Features

The pipeline is broken down into four distinct phases:

* **Phase 1: Ingestion & Chunking:** Implements a custom sliding-window text chunker with character overlap to prevent semantic severing.
* **Phase 2: Local Vector Embeddings:** Uses HuggingFace's `sentence-transformers` (`all-MiniLM-L6-v2`) to convert text chunks into standardized 384-dimensional mathematical coordinates.
* **Phase 3: In-Memory Vector Search:** Features a custom search engine utilizing NumPy to calculate Exact Nearest Neighbor (ENN) via dot-product cosine similarity.
* **Phase 4: LLM Augmentation:** Injects retrieved context into a strict prompt template and interfaces with the Google Gemini API (`gemini-2.0-flash`). Includes production-grade exponential backoff logic to handle API rate limits (HTTP 429).

## Prerequisites

* Python 3.8 or higher
* A Google Gemini API Key

## Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/MicroRAG.git](https://github.com/yourusername/MicroRAG.git)
   cd MicroRAG
   ```

2. **Install the required dependencies:**
  ```bash
  pip install -r requirements.txt
   ```


3. **Configure Environment Variables:**
Create a file named `.env` in the root directory of the project and add your Google Gemini API key:
  ```env
  GEMINI_API_KEY="your_actual_api_key_here"
   ```

## Usage

Run the main pipeline script directly from your terminal:

  ```bash
  python main.py
   ```

### Expected Output

The script will initialize the local embedding model, chunk a mock corporate knowledge base, translate the text into vectors, perform a semantic similarity search against a user query, and return a context-aware generation from the Gemini model. Execution logs will print to the console detailing each step of the data flow.

## Technologies Used

* **Python:** Core logic and object-oriented structure
* **NumPy:** High-performance matrix mathematics
* **Sentence-Transformers:** Local embedding generation
* **Google GenAI SDK:** Large Language Model integration
* **Python-dotenv:** Secure environment variable management
