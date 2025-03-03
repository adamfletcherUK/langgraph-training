"""
Memory-Augmented Workflow Pattern using LangGraph

This demo showcases a sophisticated agent that maintains both short-term conversation context
and long-term memory using vectorized storage. The agent can:
1. Remember facts from previous conversations
2. Use semantic search to find relevant memories
3. Reason about temporal relationships between memories
4. Update and maintain its knowledge base
"""

from langgraph.graph import StateGraph, END, START
import langgraph
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from uuid import uuid4

# Set tokenizer parallelism before importing HF
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Load environment variables
load_dotenv()

# Initialize models and components
llm = ChatAnthropic(model="claude-3-sonnet-20240229")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Define memory schemas


@dataclass
class Memory:
    """Represents a single memory entry"""
    id: str
    content: str
    timestamp: datetime
    context: str
    importance: float


class ConversationState(BaseModel):
    """Tracks the current state of the conversation"""
    messages: List[Dict] = Field(default_factory=list)
    current_context: str = Field(default="")
    memories: List[Memory] = Field(default_factory=list)
    last_memory_refresh: datetime = Field(default_factory=datetime.now)


class MemoryOperation(BaseModel):
    """Defines a memory operation response"""
    operation: str = Field(
        description="Type of memory operation: 'store', 'retrieve', or 'update'")
    content: str = Field(
        description="Content to store or retrieved memory content")
    importance: float = Field(description="Importance score between 0 and 1")


# Initialize vector store for long-term memory
initial_memory_id = str(uuid4())
vector_store = FAISS.from_texts(
    ["Initial memory placeholder"],
    embedding=embeddings,
    metadatas=[{
        "id": initial_memory_id,
        "timestamp": datetime.now().isoformat(),
        "context": "system",
        "importance": 0.5
    }]
)


def store_memory(content: str, context: str, importance: float) -> Memory:
    """Store a new memory in both vector store and working memory"""
    memory_id = str(uuid4())
    timestamp = datetime.now()

    # Store in vector database
    vector_store.add_texts(
        [content],
        metadatas=[{
            "id": memory_id,
            "timestamp": timestamp.isoformat(),
            "context": context,
            "importance": importance
        }]
    )

    return Memory(
        id=memory_id,
        content=content,
        timestamp=timestamp,
        context=context,
        importance=importance
    )


def retrieve_relevant_memories(query: str, k: int = 3) -> List[Memory]:
    """Retrieve relevant memories using semantic search"""
    results = vector_store.similarity_search_with_score(query, k=k)

    memories = []
    for doc, score in results:
        metadata = doc.metadata
        try:
            memories.append(Memory(
                id=metadata.get("id", str(uuid4())),  # Fallback ID if missing
                content=doc.page_content,
                timestamp=datetime.fromisoformat(metadata.get(
                    "timestamp", datetime.now().isoformat())),
                context=metadata.get("context", ""),
                importance=float(metadata.get("importance", 0.5))
            ))
        except (ValueError, KeyError) as e:
            # Skip invalid memories but continue processing
            continue

    return memories

# Define node functions for the graph


def process_user_input(state: ConversationState) -> ConversationState:
    """Process user input and retrieve relevant memories"""
    current_message = state.messages[-1]

    # Retrieve relevant memories
    memories = retrieve_relevant_memories(current_message["content"])

    # Print retrieved memories for visibility
    if memories:
        print("\n🔍 Retrieved relevant memories:")
        for mem in memories:
            print(
                f"- {mem.content} (from {mem.timestamp.strftime('%Y-%m-%d %H:%M')})")
    else:
        print("\n📝 No relevant memories found")

    # Update state with retrieved memories
    state.memories = memories
    return state


def generate_response(state: ConversationState) -> ConversationState:
    """Generate response considering conversation history and memories"""
    messages = state.messages
    memories = state.memories

    # Format memories for context
    memory_context = "\n".join([
        f"- {mem.content} (from {mem.timestamp.strftime('%Y-%m-%d %H:%M')})"
        for mem in memories
    ])

    # Create prompt with memory context
    system_prompt = f"""You are an AI assistant with both short-term and long-term memory.
    
Recent relevant memories:
{memory_context}

Use these memories when relevant to provide more contextual and personalized responses.
Always maintain a natural conversational tone.
If you use any memories in your response, explicitly mention that you're recalling them."""

    # Generate response
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *[HumanMessage(content=m["content"]) if m["type"] == "human"
          else AIMessage(content=m["content"])
          for m in messages[-5:]]  # Include last 5 messages for context
    ])

    # Store the response in conversation history
    messages.append({
        "type": "ai",
        "content": response.content
    })

    # Analyze response for potential new memories
    memory_analysis = llm.invoke([
        SystemMessage(
            content="Analyze the following conversation turn. If there are important facts or context worth remembering, extract them."),
        HumanMessage(
            content=f"User: {messages[-2]['content']}\nAssistant: {response.content}")
    ])

    # Store new memories if relevant
    if "worth remembering" in memory_analysis.content.lower():
        new_memory = store_memory(
            content=response.content,
            context=state.current_context,
            importance=0.7  # Default importance for conversation-derived memories
        )
        state.memories.append(new_memory)
        print("\n💾 Stored new memory:", response.content)

    return state


# Create the graph
workflow = StateGraph(ConversationState)

# Add nodes
workflow.add_node("process_input", process_user_input)
workflow.add_node("generate_response", generate_response)

# Add edges with proper START node connection
workflow.add_edge(START, "process_input")
workflow.add_edge("process_input", "generate_response")
workflow.add_edge("generate_response", END)

# Compile the graph
agent = workflow.compile()


def chat(message: str, context: str = "") -> str:
    """Main chat interface"""
    initial_state = ConversationState(
        messages=[{"type": "human", "content": message}],
        current_context=context,
        memories=[],
        last_memory_refresh=datetime.now()
    )

    try:
        result = agent.invoke(initial_state)
        # Access result as dictionary since it's an AddableValuesDict
        messages = result["messages"]
        for msg in reversed(messages):
            if msg.get("type") == "ai":
                return msg.get("content", "I apologize, but I couldn't generate a response.")
        return "I apologize, but I couldn't generate a response."
    except Exception as e:
        print(f"Error during chat: {str(e)}")
        return "I encountered an error while processing your request. Please try again."


if __name__ == "__main__":
    # Example usage
    print("Assistant: Hi! I'm a memory-augmented AI assistant. How can I help you today?")
    print("\nℹ️  I can remember our conversation and use it for context in future responses.")
    print("🔍 When I retrieve memories, they'll be shown with this icon")
    print("💾 When I store new memories, they'll be shown with this icon\n")

    conversation_history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            break

        response = chat(user_input)
        conversation_history.append({"type": "human", "content": user_input})
        conversation_history.append({"type": "ai", "content": response})
        print(f"Assistant: {response}")
