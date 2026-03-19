---
title: PDF RAG Chatbot
sdk: gradio
emoji: 📚
colorFrom: blue
colorTo: purple
pinned: false
sdk_version: 5.23.1
app_file: app.py
license: mit
---

# 📚 PDF RAG Chatbot

An intelligent chatbot that allows you to upload PDF documents and ask questions about their content using Retrieval-Augmented Generation (RAG).

## 🔧 Technical Workflow

### Step 1 — PDF Ingestion
- User uploads a `.pdf` file via the Gradio UI
- `PyPDF2.PdfReader` iterates through every page and extracts raw text into a single string

### Step 2 — Text Chunking
- The raw text is split into overlapping chunks of **500 words** with a **50-word overlap**
- Overlap ensures that sentences spanning chunk boundaries are not lost
- Example:
  ```
  Chunk 1 → words [0 : 500]
  Chunk 2 → words [450 : 950]   ← 50-word overlap
  Chunk 3 → words [900 : 1400]
  ```

### Step 3 — Embedding Generation
- Each chunk is passed through `sentence-transformers/all-MiniLM-L6-v2`
- The model converts every chunk into a **384-dimensional dense vector**
- These vectors numerically capture the semantic meaning of each chunk

### Step 4 — FAISS (Facebook AI Similarity Search) Vector Indexing
- All chunk embeddings are loaded into a **FAISS `IndexFlatL2`** index
- FAISS stores them in memory for fast nearest-neighbor lookup
- Both the raw chunks and their vector index are held in the `PDFChatbot` instance

### Step 5 — Query Processing
- User types a question in the chat interface
- The question is embedded using the **same SentenceTransformer model**
- FAISS performs an **L2 (Euclidean) similarity search** to find the **top-3 most semantically relevant chunks**

### Step 6 — Prompt Construction
- The 3 retrieved chunks are joined as context
- A structured prompt is built:
  ```
  Context:
  <retrieved chunks>

  Question: <user question>

  Answer:
  ```

### Step 7 — LLM Inference
- The prompt is sent to `meta-llama/Llama-3.2-1B-Instruct` via the **HuggingFace Inference API**
- Parameters: `max_tokens=500`, `temperature=0.7`
- The model generates an answer strictly grounded in the provided context
- If the answer isn't in the context, the model is instructed to say so explicitly

### Step 8 — Response Delivery
- The answer is appended to the Gradio chat history as a `(question, answer)` tuple
- The UI re-renders the chatbot component with the updated history
- The question input box is automatically cleared for the next query

---

## 🗺️ End-to-End Flow Diagram

```
┌─────────────┐
│  PDF Upload  │
└──────┬──────┘
       │ PyPDF2
       ▼
┌─────────────┐
│  Raw Text   │
└──────┬──────┘
       │ chunk_text()
       ▼
┌──────────────────┐
│  Text Chunks     │  (500 words, 50-word overlap)
└──────┬───────────┘
       │ SentenceTransformer
       ▼
┌──────────────────┐
│  Embeddings      │  (384-dim vectors)
└──────┬───────────┘
       │ FAISS IndexFlatL2
       ▼
┌──────────────────┐
│  Vector Index    │  ◄─────────────────────────┐
└──────────────────┘                             │
                                                 │
┌─────────────┐                                  │
│ User Query  │                                  │
└──────┬──────┘                                  │
       │ SentenceTransformer                     │
       ▼                                         │
┌──────────────────┐    L2 Search                │
│ Query Embedding  │ ────────────────────────────┘
└──────────────────┘
       │ Top-3 chunks
       ▼
┌──────────────────┐
│  Prompt Builder  │
└──────┬───────────┘
       │ HuggingFace Inference API
       ▼
┌──────────────────────────┐
│  Llama-3.2-1B-Instruct   │
└──────┬───────────────────┘
       │
       ▼
┌──────────────┐
│   Answer     │  → Gradio Chat UI
└──────────────┘
```

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| PDF Parsing | PyPDF2 |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS (IndexFlatL2) |
| LLM | meta-llama/Llama-3.2-1B-Instruct |
| LLM API | HuggingFace Inference API |
| UI Framework | Gradio |

---

## ⚙️ Setup & Run

```bash
# Install dependencies
pip install gradio PyPDF2 sentence-transformers faiss-cpu numpy huggingface_hub

# Set your HuggingFace API key
export HF_API_KEY=your_api_key_here

# Run the app
python app.py
```

---

## ⚠️ Known Limitations

- **In-memory only** — FAISS index is lost on server restart; no persistence layer
- **Single-user design** — one global chatbot instance; concurrent users overwrite each other's PDF
- **Text-based PDFs only** — scanned/image PDFs without OCR will yield no extractable text
- **Small LLM** — Llama 1B is fast and free-tier friendly but may struggle with complex reasoning tasks


## 🤝 Contributing

Feedback and suggestions are welcome! Feel free to open an issue or submit a pull request.

## 📄 License

This project is open-source and available under the MIT License.

---

**Built with ❤️ using HuggingFace, Gradio, and Open-Source AI models**
> Testing GitHub achievements (Pull Shark).
