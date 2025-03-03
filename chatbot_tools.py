"""
LangGraph-based Chatbot with Tool Integration

This module implements a chatbot using LangGraph that can interact with external tools.
It demonstrates how to create a graph-based conversational AI system with the ability
to use tools for enhanced functionality.

Key Components:
- State Management: Uses TypedDict for maintaining conversation state
- Tool Integration: Incorporates Tavily search as an external tool
- Graph Structure: Implements a cyclic graph for handling conversations and tool usage

Author: [Your Name]
Date: [Current Date]
"""

from IPython.display import Image, display
from langchain_core.messages import ToolMessage
import json
import os
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults

# Configuration and Setup
# ----------------------


def check_environment():
    """Verify required environment variables are set."""
    if not os.getenv("TAVILY_API_KEY"):
        raise ValueError(
            "Please set the TAVILY_API_KEY environment variable. "
            "You can get one at https://tavily.com/#api"
        )


check_environment()

# Tool Setup
# ----------


def setup_tools():
    """Initialize and configure available tools."""
    search_tool = TavilySearchResults(max_results=2)
    return [search_tool]


tools = setup_tools()

# State Management
# ---------------


class State(TypedDict):
    """
    Represents the conversation state.

    Attributes:
        messages: List of conversation messages with automatic message addition handling
    """
    messages: Annotated[list, add_messages]

# LLM Configuration
# ----------------


def setup_llm():
    """Configure the language model with tool integration."""
    base_llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
    return base_llm.bind_tools(tools)


llm_with_tools = setup_llm()

# Node Definitions
# --------------


def chatbot(state: State):
    """
    Main chatbot node that processes user input and generates responses.

    Args:
        state: Current conversation state

    Returns:
        dict: Updated state with new message
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


class BasicToolNode:
    """
    Node responsible for executing tools requested by the chatbot.

    This node processes tool calls from the LLM and returns their results
    back to the conversation flow.
    """

    def __init__(self, tools: list) -> None:
        """
        Initialize the tool node with available tools.

        Args:
            tools: List of available tools
        """
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        """
        Execute requested tools and return their results.

        Args:
            inputs: Dictionary containing messages with tool requests

        Returns:
            dict: Tool execution results
        """
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

# Graph Construction
# ----------------


def build_graph():
    """
    Construct the conversation flow graph.

    Returns:
        StateGraph: Compiled graph ready for execution
    """
    graph_builder = StateGraph(State)

    # Add nodes
    graph_builder.add_node("chatbot", chatbot)
    tool_node = BasicToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)

    # Add edges with routing
    def route_tools(state: State):
        """Route to tools node if tool calls present, otherwise end."""
        if isinstance(state, list):
            ai_message = state[-1]
        elif messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state: {state}")

        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return END

    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        {"tools": "tools", END: END},
    )

    # Complete the graph
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    return graph_builder.compile()

# Main Execution
# -------------


def stream_graph_updates(user_input: str, graph):
    """
    Stream updates from the graph execution.

    Args:
        user_input: User's message
        graph: Compiled conversation graph
    """
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


def main():
    """Main execution loop for the chatbot."""
    graph = build_graph()

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(user_input, graph)
        except Exception as e:
            # Fallback for environments where input() is not available
            print("Error:", str(e))
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input, graph)
            break


if __name__ == "__main__":
    main()
