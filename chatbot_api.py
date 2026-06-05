import os
import json
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str

class ChatbotSystem:
    def __init__(self):
        # Lazy imports — only loaded when first chat request comes in,
        # so server startup is never blocked by missing packages.
        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer
            from mistralai import Mistral
        except ImportError as e:
            raise RuntimeError(
                f"Chatbot dependency not installed: {e}. "
                "Run: pip install faiss-cpu sentence-transformers mistralai"
            )

        # Store numpy ref for use in methods
        self._np = np
        self._faiss = faiss

        MISTRAL_API_KEY = "eol9f2CNSbekOhpB0mb62x20jvyhNnGo"
        self.client = Mistral(api_key=MISTRAL_API_KEY)

        # Resolve healthcare.json relative to this script's location
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(base_dir, "healthcare.json")

        self.data = self._load_data()
        self.docs = self._create_documents(self.data)

        print("[Chatbot] Loading sentence-transformer model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = self._build_vector_store(self.docs)
        print(f"[Chatbot] Ready — {len(self.data)} health conditions loaded.")

    def _load_data(self):
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Chatbot] Error loading healthcare data: {e}")
            return []

    def _create_documents(self, data):
        docs = []
        for item in data:
            symptoms = ", ".join(item.get("symptoms", []))
            treatments = ", ".join(item.get("treatment", []))
            docs.append(
                f"Disease: {item.get('name', 'Unknown')}\n"
                f"Symptoms: {symptoms}\n"
                f"Home Treatment: {treatments}"
            )
        return docs

    def _build_vector_store(self, docs):
        if not docs:
            return None
        np = self._np
        faiss = self._faiss
        embeddings = self.embedder.encode(docs)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))
        return index

    def retrieve_context(self, query, k=3):
        if not self.index:
            return ""
        np = self._np
        q_emb = self.embedder.encode([query])
        _, idx = self.index.search(np.array(q_emb), k)
        return "\n\n".join([self.docs[i] for i in idx[0]])

    def generate_response(self, query: str, model_name="mistral-large-latest"):
        context = self.retrieve_context(query)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a friendly healthcare assistant on the SignLink platform. "
                    "Answer ONLY using the given context. "
                    "If the issue seems serious, advise consulting a doctor."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]
        try:
            response = self.client.chat.complete(
                model=model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling Mistral API: {str(e)}"


# Global singleton — initialized on first request, not at import time
_chatbot_system = None

def get_chatbot_response(query: str) -> str:
    global _chatbot_system
    if _chatbot_system is None:
        print("[Chatbot] Initializing for first time...")
        _chatbot_system = ChatbotSystem()
    return _chatbot_system.generate_response(query)
