from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import tempfile
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize ChromaDB
def get_vector_collection():
    """Gets or creates ChromaDB collection with Ollama embeddings"""
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )
    chroma_client = chromadb.PersistentClient(path="./demo-ai")
    return chroma_client.get_or_create_collection(
        name="ai-project",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
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



# Add to your existing imports
class DocumentChunk(BaseModel):
    page_content: str
    metadata: dict

class AddToVectorRequest(BaseModel):
    chunks: List[DocumentChunk]
    file_name: str

# Add this endpoint to your existing FastAPI app
@app.post("/add-to-vector-collection")
async def add_to_vector_collection(request: AddToVectorRequest):
    """Adds document splits to ChromaDB vector collection"""
    try:
        collection = get_vector_collection()
        documents = []
        metadatas = []
        ids = []
        
        for idx, chunk in enumerate(request.chunks):
            documents.append(chunk.page_content)
            metadatas.append(chunk.metadata)
            ids.append(f"{request.file_name}_{idx}")
        
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        return {"status": "success", "message": "Data added to vector store", "count": len(documents)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





# Add to your existing models
class QueryRequest(BaseModel):
    prompt: str
    n_results: int = 10

# Add this endpoint to your existing FastAPI app
@app.post("/query-collection")
async def query_collection(request: QueryRequest):
    """Queries the vector collection for relevant documents"""
    try:
        collection = get_vector_collection()
        results = collection.query(
            query_texts=[request.prompt],
            n_results=request.n_results
        )
        
        # Format results for better JSON serialization
        # formatted_results = {
        #     "documents": results["documents"][0] if results["documents"] else [],
        #     "metadatas": results["metadatas"][0] if results["metadatas"] else [],
        #     "ids": results["ids"][0] if results["ids"] else [],
        #     "distances": results["distances"][0] if results["distances"] else []
        # }
        
        return {
            "status": "success",
            "results": results,
            "query": request.prompt,
            #"count": len(formatted_results["documents"])
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)