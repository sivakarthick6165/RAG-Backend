import os
import httpx
from langchain_community.llms import Ollama
from .vector_store import VectorStoreManager
from dotenv import load_dotenv

load_dotenv()

class RAGPipeline:
    def __init__(self):
        self.vector_store_manager = VectorStoreManager()
        self.base_url = os.getenv("BASE_URL", "http://localhost:11434").rstrip("/")
        self.default_model = os.getenv("MODEL", "llama2")
        self.api_key = os.getenv("API_KEY")

    async def get_response(self, question: str, model: str = None, filename: str = None):
        if model is None:
            model = self.default_model

        # 1. Retrieve relevant chunks
        filter = {"filename": filename} if filename else None
        docs = self.vector_store_manager.similarity_search(question, k=5, filter=filter)
        if not docs:
            return {
                "answer": "I'm sorry, I can only answer questions based on the uploaded documents.",
                "sources": []
            }
        context = "\n\n".join([doc.page_content for doc in docs])

        # 2. Build prompt
        prompt = f"""
        TRANSCRIPT OF AN INTERNAL DOCUMENT:
        {context}

        ---
        INSTRUCTIONS:
        1. Answer the Question below ONLY using the provided TRANSCRIPT above.
        2. If the answer is not contained in the TRANSCRIPT, state exactly: "I'm sorry, I can only answer questions based on the uploaded documents."
        3. Do NOT use your own external knowledge.
        4. Be precise and literal.

        Question: {question}
        Answer:
        """

        # 3. Call Ollama via Chat API for better instruction adherence
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    headers=headers,
                    json={
                        "model": model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a restricted RAG assistant. You ONLY use the provided text to answer. You have NO external knowledge. If the answer isn't in the text, you MUST say 'I'm sorry, I can only answer questions based on the uploaded documents.' Never summarize your own knowledge."
                            },
                            {
                                "role": "user",
                                "content": f"DOCUMENT TEXT:\n{context}\n\nQUESTION: {question}"
                            }
                        ],
                        "stream": False,
                        "options": {
                            "temperature": 0.0,
                            "num_predict": 512
                        }
                    },
                    timeout=120.0 
                )
                response.raise_for_status()
                result = response.json()
                # Chat API returns 'message': {'content': '...'}
                answer = result.get("message", {}).get("content", "")
                return {
                    "answer": answer,
                    "sources": [doc.metadata for doc in docs]
                }
            except httpx.ConnectError:
                return {
                    "error": f"Could not connect to Ollama at {self.base_url}.",
                    "answer": "Connection Error."
                }
            except Exception as e:
                error_detail = str(e)
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_detail = e.response.json().get('error', e.response.text)
                    except:
                        error_detail = e.response.text
                
                return {
                    "error": f"Ollama Error: {error_detail}",
                    "answer": "I encountered an error while trying to generate an answer."
                }

    async def get_available_models(self):
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/api/tags",
                    headers=headers
                )
                response.raise_for_status()
                models_data = response.json()
                # Filter out models that are likely embedding-only
                return [
                    m["name"] for m in models_data.get("models", [])
                    if "embed" not in m["name"].lower()
                ]
            except Exception as e:
                return []
