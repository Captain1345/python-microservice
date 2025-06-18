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

from pinecone import Pinecone
# Add this import at the top with your other imports
from llm_utils import call_llm_with_history, Message, call_llm_for_feedback


app = FastAPI()
load_dotenv()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_vector_collection():
    """Gets or creates Pinecone index with Ollama embeddings"""

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    index_name = "better-pm"
    
    if not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "chunk_text"}
            }
        )
    
    index = pc.Index(index_name)

    return index


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

@app.post("/add-to-vector-collection")
async def add_to_vector_collection(request: AddToVectorRequest):
    """Adds document splits to Pinecone vector collection"""
    try:
        index = get_vector_collection()
        documents = []
        # Process chunks in batches to avoid overwhelming the API
        batch_size = 90
        total_chunks = len(request.chunks)
        for idx, chunk in enumerate(request.chunks):
             documents.append({
                "id": f"{request.file_name}_{idx}",
                "text": chunk.page_content,
                })
        # Upsert in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            index.upsert_records(
                namespace="pm-interviews",
                records=batch)
        
        
        return {"status": "success", "message": "Data added to Pinecone", "count": total_chunks}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class QueryRequest(BaseModel):
    conversationHistory: List[Message] = []  # History from Supabase
    lastMessageSent: str
    n_results: int = 10

@app.post("/query-collection")
async def query_collection(request: QueryRequest):
    """Queries the vector collection for relevant documents"""
    try:
        index = get_vector_collection()
        
        query_payload = {
            "inputs": {
                "text": request.lastMessageSent
            },
            "top_k": request.n_results,
        }
        # Query Pinecone
        query_results = index.search(
            query=query_payload,
            namespace="pm-interviews"
        )
        
        if not query_results.get('result', {}).get('hits'):
            raise HTTPException(status_code=404, detail="No results found")
        # Extract documents from results
        documents = [hit["fields"]["text"] for hit in query_results["result"]["hits"]]
        
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
            "query": request.lastMessageSent,
            "llmResponse": response
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class FeedbackRequest(BaseModel):
    conversation_id: str
    allMessages: List[Message]


@app.post("/pm-feedback")
async def get_pm_feedback(request: FeedbackRequest):
    """
    Receives a conversation and returns feedback from a Gemini LLM.
    """
    try:
        # This prompt instructs the LLM to act as a PM and provide feedback
        feedback_prompt = (
            """You are a world-class product manager. Please analyze the following interview transcript and give your
            insightful, actionable feedback"""
        )

        # The 'call_llm_for_feedback' function expects a prompt and a history.
        # We'll use our feedback_prompt as the main prompt and pass all messages as history.
        # No external context from the vector store is needed here.
        response = call_llm_for_feedback(
            feedback_prompt=feedback_prompt,
            all_messages=request.allMessages
        )

        return {
            "status": "success",
            "conversation_id": request.conversation_id,
            "llmFeedback": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
