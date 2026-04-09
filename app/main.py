import os
import shutil
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from .db.database import engine, get_db, Base
from .models import models
from .services.parser import FileParser
from .services.chunker import TextChunker
from .rag.pipeline import RAGPipeline
from .rag.vector_store import VectorStoreManager
from dotenv import load_dotenv

load_dotenv()

# Create tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Full Stack RAG System API")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
parser = FileParser()
chunker = TextChunker()
vector_store = VectorStoreManager()
rag_pipeline = RAGPipeline()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    # 1. Save file locally
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Add to database
    db_doc = models.Document(
        filename=file.filename,
        file_type=file.filename.split('.')[-1]
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)

    try:
        # 3. Parse file
        text = parser.extract_text(file_path)
        
        # 4. Chunk text
        chunks = chunker.chunk_text(text)
        
        # 5. Store in FAISS and DB
        chunk_metadatas = []
        for i, chunk_text in enumerate(chunks):
            db_chunk = models.Chunk(
                document_id=db_doc.id,
                chunk_text=chunk_text,
                embedding_id=f"{db_doc.id}_{i}"
            )
            db.add(db_chunk)
            chunk_metadatas.append({"doc_id": db_doc.id, "filename": db_doc.filename, "chunk_index": i})
        
        db.commit()
        
        # 6. Add to Vector Store
        vector_store.add_texts(chunks, metadatas=chunk_metadatas)

        return {"status": "success", "filename": file.filename, "document_id": db_doc.id, "chunks": len(chunks)}
    except Exception as e:
        db.delete(db_doc)
        db.commit()
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_rag(
    question: str = Body(..., embed=True),
    model: Optional[str] = Body(None, embed=True),
    filename: Optional[str] = Body(None, embed=True)
):
    result = await rag_pipeline.get_response(question, model, filename)
    return result

@app.get("/models")
async def list_models():
    models_list = await rag_pipeline.get_available_models()
    return {"models": models_list}

@app.get("/documents")
async def list_documents(db: Session = Depends(get_db)):
    docs = db.query(models.Document).all()
    return docs

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: int, db: Session = Depends(get_db)):
    doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # In a real app, we should also remove from FAISS index. 
    # FAISS deletion is tricky with LangChain's basic wrapper. 
    # For now, we'll just delete from DB.
    db.delete(doc)
    db.commit()
    return {"status": "success", "message": f"Document {doc_id} deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
