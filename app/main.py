import os
import shutil
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from .db.database import engine, get_db, Base
from .models import models
from .services.parser import FileParser
from .services.chunker import TextChunker
from .rag.pipeline import RAGPipeline
from .rag.vector_store import VectorStoreManager

load_dotenv()

# Base directory (IMPORTANT for Railway)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

# Upload directory fix
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- API ROUTES ---------------- #

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    db_doc = models.Document(
        filename=file.filename,
        file_type=file.filename.split('.')[-1]
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)

    try:
        text = parser.extract_text(file_path)
        chunks = chunker.chunk_text(text)

        chunk_metadatas = []
        for i, chunk_text in enumerate(chunks):
            db_chunk = models.Chunk(
                document_id=db_doc.id,
                chunk_text=chunk_text,
                embedding_id=f"{db_doc.id}_{i}"
            )
            db.add(db_chunk)

            chunk_metadatas.append({
                "doc_id": db_doc.id,
                "filename": db_doc.filename,
                "chunk_index": i
            })

        db.commit()

        vector_store.add_texts(chunks, metadatas=chunk_metadatas)

        return {
            "status": "success",
            "filename": file.filename,
            "document_id": db_doc.id,
            "chunks": len(chunks)
        }

    except Exception as e:
        db.delete(db_doc)
        db.commit()
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
async def query_rag(
    question: str = Body(..., embed=True),
    model: Optional[str] = Body(None, embed=True),
    filename: Optional[str] = Body(None, embed=True)
):
    result = await rag_pipeline.get_response(question, model, filename)
    return result


@app.get("/api/models")
async def list_models():
    models_list = await rag_pipeline.get_available_models()
    return {"models": models_list}


@app.get("/api/documents")
async def list_documents(db: Session = Depends(get_db)):
    return db.query(models.Document).all()


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: int, db: Session = Depends(get_db)):
    doc = db.query(models.Document).filter(models.Document.id == doc_id).first()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    db.delete(doc)
    db.commit()

    return {
        "status": "success",
        "message": f"Document {doc_id} deleted"
    }

# ---------------- FRONTEND (React) ---------------- #

# Serve React static files
app.mount(
    "/", 
    StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), 
    name="static"
)

# React SPA fallback (fix refresh issue)
@app.get("/{full_path:path}")
async def serve_react(full_path: str):
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


# ---------------- LOCAL RUN ---------------- #

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)