from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import tempfile
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProcessedDocument(BaseModel):
    page_content: str
    metadata: Dict
    file_name: str  # Track which file the chunk came from

@app.post("/convert-pdfs-chunks", response_model=List[ProcessedDocument])
async def process_pdfs(files: List[UploadFile] = File(...)):
    all_chunks = []
    temp_files = []
    
    try:
        for file in files:
            # Create temp file for each PDF
            temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
            temp_file.write(await file.read())
            temp_path = temp_file.name
            temp_files.append(temp_path)
            temp_file.close()

            # Process each PDF
            loader = PyMuPDFLoader(temp_path)
            docs = loader.load()

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=100,
                separators=["\n\n", "\n", ".", "?", "!", " ", ""],
            )
            chunks = text_splitter.split_documents(docs)

            # Add chunks with file name tracking
            for chunk in chunks:
                all_chunks.append({
                    "page_content": chunk.page_content,
                    "metadata": {**chunk.metadata, "source_file": file.filename},
                    "file_name": file.filename
                })

        return all_chunks

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up all temp files
        for temp_path in temp_files:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)