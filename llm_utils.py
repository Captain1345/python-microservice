from typing import List
import os
import ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel

# Import the system prompt
from prompts import system_prompt

# Define the Message class here to avoid circular imports
class Message(BaseModel):
    id: str = None
    conversation_id: str = None
    role: str  # "user" or "assistant"
    content: str
    created_at: str = None
    metadata: dict = {}


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



def truncate_conversation_history(history, max_messages=10):
    """Truncate conversation history to avoid exceeding context limits"""
    if len(history) <= max_messages:
        return history
    
    # Keep the most recent messages
    return history[-max_messages:]


def call_llm_with_history(context: str, prompt: str, conversation_history: List[Message] = []):
    """Calls the Gemini model with context, prompt, and conversation history.

    Args:
        context: String containing the relevant context for answering the question
        prompt: String containing the user's current question
        conversation_history: List of previous messages from Supabase

    Returns:
        String containing the generated response
    """
    try:
        # Initialize the Gemini chat model
        chat = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-05-20",
            temperature=1,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            streaming=True,
            convert_system_message_to_human=False
        )

        # Convert conversation history to LangChain message format
        messages = [SystemMessage(content=system_prompt)]
        
        # Add conversation history
        for msg in conversation_history:
            if msg.role.lower() == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role.lower() == "assistant":
                messages.append(AIMessage(content=msg.content))
        
        # Add the current context and question
        messages.append(HumanMessage(content=f"Context: {context}\nQuestion: {prompt}"))

        # Get streaming response
        response = ""
        for chunk in chat.stream(messages):
            response += chunk.content

        return response

    except Exception as e:
        raise Exception(f"Error calling Gemini API: {str(e)}")