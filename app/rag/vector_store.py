import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangchainDocument
from dotenv import load_dotenv

load_dotenv()

class VectorStoreManager:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            base_url=os.getenv("BASE_URL", "http://localhost:11434").rstrip("/"),
            model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        )
        self.index_path = os.getenv("FAISS_INDEX_PATH", "./faiss_index")
        self.vector_store = None
        self._load_or_create_index()

    def _load_or_create_index(self):
        if os.path.exists(os.path.join(self.index_path, "index.faiss")):
            self.vector_store = FAISS.load_local(
                self.index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            # Create an empty index if not exists
            # FAISS needs at least one document to initialize, 
            # so we'll handle lazy initialization in add_texts
            self.vector_store = None

    def add_texts(self, texts: list[str], metadatas: list[dict] = None):
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        else:
            self.vector_store.add_texts(texts, metadatas=metadatas)
        self.save_index()

    def save_index(self):
        if self.vector_store:
            self.vector_store.save_local(self.index_path)

    def similarity_search(self, query: str, k: int = 4, filter: dict = None):
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search(query, k=k, filter=filter)
