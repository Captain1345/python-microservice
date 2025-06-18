from typing import List
import os
import ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel
import json
from langchain_core.prompts import ChatPromptTemplate

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
        # Handle potential exceptions, e.g., API errors
        print(f"An error occurred: {e}")
        return None


def call_llm_for_feedback(feedback_prompt: str, all_messages: List[Message]):
    """
    Calls the Gemini LLM with a formatted interview transcript to get feedback.

    Args:
        feedback_prompt (str): The prompt instructing the LLM on how to provide feedback.
        all_messages (List[Message]): The list of all messages in the conversation.

    Returns:
        str: The generated feedback from the LLM.
    """
    try:
        # Initialize the Gemini chat model
        chat = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-05-20",
            temperature=0.7,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            convert_system_message_to_human=True
        )

        # Format the conversation history into a structured transcript
        interview_transcript = [{"role": msg.role, "content": msg.content} for msg in all_messages]

        # Use LangChain's prompt templates for a clean and maintainable prompt
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", feedback_prompt),
                ("human", "Please provide feedback on the following interview transcript:\n\n```json\n{transcript}\n```"),
            ]
        )

        # Create the LangChain Expression Language (LCEL) chain
        chain = prompt_template | chat

        # For debugging: Print the formatted JSON transcript
        formatted_transcript = json.dumps(interview_transcript, indent=2)
        print("--- Sending the following transcript to the LLM ---")
        print(formatted_transcript)
        print("---------------------------------------------------")

        # Invoke the chain with the interview transcript
        response = chain.invoke({"transcript": formatted_transcript})

        return response.content

    except Exception as e:
        print(f"An error occurred during feedback generation: {e}")
        return None