import gradio as gr
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from huggingface_hub import InferenceClient
import os

# Initialize the embedding model (lightweight and efficient)
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize HuggingFace Inference Client for LLM
# Uses Inference API - API key needed
client = InferenceClient(api_key=os.getenv("HF_API_KEY"))

class PDFChatbot:
    def __init__(self):
        self.chunks = []
        self.embeddings = None
        self.index = None
        
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    
    def chunk_text(self, text, chunk_size=500, overlap=50):
        """Split text into overlapping chunks for better context"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
    
    def create_vector_store(self, chunks):
        """Create FAISS vector store from text chunks"""
        self.chunks = chunks
        self.embeddings = embedding_model.encode(chunks)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(self.embeddings).astype('float32'))
        
    def retrieve_relevant_chunks(self, query, top_k=3):
        """Retrieve most relevant chunks for the query"""
        query_embedding = embedding_model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        
        relevant_chunks = [self.chunks[i] for i in indices[0]]
        return relevant_chunks
    
    def generate_answer(self, query, context):
        """Generate answer using LLM with retrieved context"""
        prompt = f"""Based on the following context, answer the question. 
    If the answer is not in the context, say "I cannot find this information in the document."
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:"""
    
        try:
            response = client.chat_completion(
                model="meta-llama/Llama-3.2-1B-Instruct",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7,
            )
    
            # Extract assistant output
            return response.choices[0].message["content"]
    
        except Exception as e:
            return f"Error generating response: {str(e)}"


# Initialize chatbot
chatbot = PDFChatbot()

def process_pdf(pdf_file):
    try:
        if pdf_file is None:
            return "Please upload a PDF file first."

        text = chatbot.extract_text_from_pdf(pdf_file)
        if not text or text.strip() == "":
            return "Error: No extractable text found in this PDF."

        chunks = chatbot.chunk_text(text)
        if len(chunks) == 0:
            return "Error: Could not create text chunks."

        chatbot.create_vector_store(chunks)
        return f"✅ PDF processed successfully! Found {len(chunks)} text chunks."

    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"

def answer_question(question, chat_history):
    try:
        if not chatbot.chunks:
            msg = "Please upload and process a PDF first."
            chat_history.append(("", msg))
            return chat_history, chat_history

        if not question.strip():
            return chat_history, chat_history

        relevant_chunks = chatbot.retrieve_relevant_chunks(question)
        context = "\n\n".join(relevant_chunks)

        answer = chatbot.generate_answer(question, context)
        chat_history.append((question, answer))
        return chat_history, chat_history

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        chat_history.append(("", error_msg))
        return chat_history, chat_history


def clear_chat():
    """Clear chat history"""
    return [], []

# Create Gradio interface
with gr.Blocks(title="PDF RAG Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📚 PDF RAG Chatbot
        
        Upload a PDF document and ask questions about its content. This chatbot uses Retrieval-Augmented Generation (RAG) 
        to provide accurate answers based on your document.
        
        **How it works:**
        1. Upload your PDF file
        2. Click "Process PDF" to analyze the document
        3. Ask questions in the chat interface
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(
                label="Upload PDF Document",
                file_types=[".pdf"],
                type="filepath"
            )
            process_btn = gr.Button("📄 Process PDF", variant="primary")
            status_output = gr.Textbox(
                label="Status",
                lines=3,
                interactive=False
            )
            
            gr.Markdown("### ℹ️ Tips")
            gr.Markdown(
                """
                - Works best with text-based PDFs
                - Larger documents may take longer to process
                - Ask specific questions for better answers
                - Try questions like:
                  - "What is the main topic?"
                  - "Summarize the key findings"
                  - "What are the conclusions?"
                """
            )
        
        with gr.Column(scale=2):
            chatbot_interface = gr.Chatbot(
                label="Chat with your PDF",
                height=400
            )
            
            with gr.Row():
                question_input = gr.Textbox(
                    label="Ask a question",
                    placeholder="What is this document about?",
                    scale=4
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            
            clear_btn = gr.Button("Clear Chat")
    
    # Store chat history
    chat_state = gr.State([])
    
    # Event handlers
    process_btn.click(
        fn=process_pdf,
        inputs=[pdf_input],
        outputs=[status_output]
    )
    
    submit_btn.click(
        fn=answer_question,
        inputs=[question_input, chat_state],
        outputs=[chatbot_interface, chat_state]
    ).then(
        lambda: "",
        outputs=[question_input]
    )
    
    question_input.submit(
        fn=answer_question,
        inputs=[question_input, chat_state],
        outputs=[chatbot_interface, chat_state]
    ).then(
        lambda: "",
        outputs=[question_input]
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot_interface, chat_state]
    )
    
    gr.Markdown(
        """
        ---
        ### 🔧 Technical Stack
        - **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
        - **Vector Store**: FAISS
        - **LLM**: Llama-3.2-1B-Instruct (via HuggingFace Inference API)
        - **Framework**: Gradio
        
        ---
        💡 **Built to demonstrate practical RAG implementation**  
        ⭐ If you find this useful, please give it a like!
        """
    )

if __name__ == "__main__":
    demo.launch(share=False)
