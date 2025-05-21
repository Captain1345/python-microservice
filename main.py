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
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

import ollama
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from prompts import system_prompt


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


def call_local_llm(context: str, prompt: str):
    """Calls the language model with context and prompt to generate a response.

    Uses Ollama to stream responses from a language model by providing context and a
    question prompt. The model uses a system prompt to format and ground its responses appropriately.

    Args:
        context: String containing the relevant context for answering the question
        prompt: String containing the user's question

    Yields:
        String chunks of the generated response as they become available from the model

    Raises:
        OllamaError: If there are issues communicating with the Ollama API
    """
    response = ollama.chat(
        model="llama3.2:latest",
        stream=True,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Context: {context}, Question: {prompt}",
            },
        ],
    )
    full_response = ""
    for chunk in response:
        if chunk["done"] is False:
            full_response += chunk["message"]["content"]
    
    return full_response



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
        print("RESULTS: ", results)
        context = results.get("documents")[0]
        relevant_text, relevant_text_ids = re_rank_cross_encoders(documents=context, prompt=request.prompt)
   
        prompt=request.prompt
        response = call_llm(context=relevant_text, prompt=prompt)
        return {
            "status": "success",
            "results": results,
            "query": request.prompt,
            "llmResponse": response
            #"count": len(formatted_results["documents"])
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def call_llm(context: str, prompt: str):
    """Calls the Gemini model through LangChain with context and prompt to generate a response.

    Args:
        context: String containing the relevant context for answering the question
        prompt: String containing the user's question

    Returns:
        String containing the generated response

    Raises:
        Exception: If there are issues communicating with the Gemini API
    """
    try:
        # Initialize the Gemini chat model
        chat = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.7,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            streaming=True,
            convert_system_message_to_human=True
        )

        # Create messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context: {context}\nQuestion: {prompt}")
        ]

        # Get streaming response
        response = ""
        for chunk in chat.stream(messages):
            response += chunk.content

        return response

    except Exception as e:
        raise Exception(f"Error calling Gemini API: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)