from tkinter import FALSE
from dotenv import load_dotenv
import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import tempfile
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from langchain_ollama import OllamaEmbeddings

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
# Add this import at the top with your other imports
from llm_utils import call_local_llm, call_llm_with_history, truncate_conversation_history, Message


app = FastAPI()
load_dotenv()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize ChromaDB
# def get_vector_collection():
#     """Gets or creates ChromaDB collection with Ollama embeddings"""
#     ollama_ef = OllamaEmbeddingFunction(
#         url="http://localhost:11434/api/embeddings",
#         model_name="nomic-embed-text:latest",
#     )
#     chroma_client = chromadb.PersistentClient(path="./demo-ai")
#     return chroma_client.get_or_create_collection(
#         name="ai-project",
#         embedding_function=ollama_ef,
#         metadata={"hnsw:space": "cosine"},
#     )

def get_vector_collection():
    """Gets or creates Pinecone index with Ollama embeddings"""
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Define index name
    index_name = "nomic-index"
    
    # Check if index already exists
    # if index_name not in pinecone.list_indexes():
    #     # Create index if it doesn't exist
    #     pinecone.create_index(
    #         name=index_name,
    #         dimension=384,  # Dimension for nomic-embed-text model
    #         metric="cosine"
    #     )
    
    # Connect to the index
    index = pc.Index(index_name)
    
    # Create embeddings function
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest",
    )
        
    #embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)
    
    return index, embeddings


def re_rank_cross_encoders(prompt: str, documents: list[str]) -> tuple[str, list[int]]:
    """Re-ranks documents using a cross-encoder model for more accurate relevance scoring.

    Uses the MS MARCO MiniLM cross-encoder model to re-rank the input documents based on
    their relevance to the query prompt. Returns the concatenated text of the top 3 most
    relevant documents along with their indices.

    Args:
        documents: List of document strings to be re-ranked.

    Returns:
        tuple: A tuple containing:
            - relevant_text (str): Concatenated text from the top 3 ranked documents
            - relevant_text_ids (list[int]): List of indices for the top ranked documents

    Raises:
        ValueError: If documents list is empty
        RuntimeError: If cross-encoder model fails to load or rank documents
    """
    relevant_text = ""
    relevant_text_ids = []
    """ This is a lightweight cross-encoder model. when moving to production, change to 
    high performing model """
    encoder_model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
    ranks = encoder_model.rank(prompt, documents, top_k=5)
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevant_text_ids.append(rank["corpus_id"])

    return relevant_text, relevant_text_ids

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
                chunk_size=800,
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
# @app.post("/add-to-vector-collection")
# async def add_to_vector_collection(request: AddToVectorRequest):
#     """Adds document splits to ChromaDB vector collection"""
#     try:
#         collection = get_vector_collection()
#         documents = []
#         metadatas = []
#         ids = []
        
#         for idx, chunk in enumerate(request.chunks):
#             documents.append(chunk.page_content)
#             metadatas.append(chunk.metadata)
#             ids.append(f"{request.file_name}_{idx}")
        
#         collection.upsert(
#             documents=documents,
#             metadatas=metadatas,
#             ids=ids
#         )
        
#         return {"status": "success", "message": "Data added to vector store", "count": len(documents)}
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-to-vector-collection")
async def add_to_vector_collection(request: AddToVectorRequest):
    """Adds document splits to Pinecone vector collection"""
    try:
        index, embeddings = get_vector_collection()
        
        # Process chunks in batches to avoid overwhelming the API
        batch_size = 100
        total_chunks = len(request.chunks)
        
        for i in range(0, total_chunks, batch_size):
            batch_chunks = request.chunks[i:i+batch_size]
            vectors_to_upsert = []
            
            for idx, chunk in enumerate(batch_chunks):
                # Generate embedding for the document
                embedding = embeddings.embed_query(chunk.page_content)
                
                # Create a unique ID for this chunk
                unique_id = f"{request.file_name}_{i+idx}"
                
                # Prepare vector for upsert
                vectors_to_upsert.append({
                    "id": unique_id,
                    "values": embedding,
                    "metadata": {
                        **chunk.metadata,
                        "text": chunk.page_content  # Store the text in metadata for retrieval
                    }
                })
            
            # Upsert vectors to Pinecone
            index.upsert(vectors=vectors_to_upsert)
        
        return {"status": "success", "message": "Data added to Pinecone", "count": total_chunks}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class QueryRequest(BaseModel):
    conversationHistory: List[Message] = []  # History from Supabase
    lastMessageSent: str
    n_results: int = 10


# Add this endpoint to your existing FastAPI app
# @app.post("/query-collection")
# async def query_collection(request: QueryRequest):
#     """Queries the vector collection for relevant documents"""
#     try:
#         collection = get_vector_collection()
#         results = collection.query(
#             query_texts=[request.lastMessageSent],
#             n_results=request.n_results
#         )
#         context = results.get("documents")[0]
#         relevant_text, relevant_text_ids = re_rank_cross_encoders(prompt=request.lastMessageSent, documents=context)
   
#             # Pass conversation history to the LLM
#         response = call_llm_with_history(
#             context=relevant_text, 
#             prompt=request.lastMessageSent,
#             conversation_history=request.conversationHistory
#         )
#         return {
#             "status": "success",
#             "results": results,
#             "query": request.lastMessageSent,
#             "llmResponse": response
#             #"count": len(formatted_results["documents"])
#         }
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
@app.post("/query-collection")
async def query_collection(request: QueryRequest):
    """Queries the vector collection for relevant documents"""
    try:
        index, embeddings = get_vector_collection()
        
        # Generate embedding for the query
        query_embedding = embeddings.embed_query(request.lastMessageSent)
        
        # Query Pinecone
        query_results = index.query(
            vector=query_embedding,
            top_k=request.n_results,
            include_metadata=True
        )
        
        # Extract documents from results
        documents = [match["metadata"]["text"] for match in query_results["matches"]]
        
        # Re-rank documents
        relevant_text, relevant_text_ids = re_rank_cross_encoders(
            prompt=request.lastMessageSent, 
            documents=documents
        )
   
        # Pass conversation history to the LLM
        response = call_llm_with_history(
            context=relevant_text, 
            prompt=request.lastMessageSent,
            conversation_history=request.conversationHistory
        )
        
        return {
            "status": "success",
            #"results": query_results["matches"],
            "query": request.lastMessageSent,
            "llmResponse": response
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
