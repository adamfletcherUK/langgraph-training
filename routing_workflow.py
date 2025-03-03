"""
LangGraph Agent Routing Tutorial
===============================

This script demonstrates how to create a multi-agent routing system using LangGraph.
It shows how to:
1. Define specialized agents for different tasks
2. Create a router to direct queries to appropriate agents
3. Manage state between agents
4. Set up conditional routing logic
5. Build and compile a workflow graph

Key Concepts:
- StateGraph: The main workflow container that manages state transitions
- MessagesState: TypedDict that defines our workflow's state structure
- Conditional Edges: Rules that determine how queries flow between agents
"""

from typing import Annotated, Any, Dict, TypedDict, List, Literal, Union
import os
import json
import math
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatAnthropic(model="claude-3-haiku-20240307")

# Define our state type


class MessagesState(TypedDict):
    """
    Defines the structure of our workflow state.

    Attributes:
        messages (List[BaseMessage]): History of messages in the conversation
        next_agent (str | None): The next agent to route to (set by router)
        final_answer (str | None): The final response from an agent
        context (Dict[str, Any]): Additional context data shared between agents
    """
    messages: List[BaseMessage]
    next_agent: str | None
    final_answer: str | None
    context: Dict[str, Any]

# Tools for our agents


@tool
def calculate_expression(expression: str) -> str:
    """Safely evaluate a mathematical expression and return the result."""
    try:
        allowed_names = {
            "abs": abs, "float": float, "int": int,
            "max": max, "min": min, "pow": pow,
            "round": round, "sum": sum
        }
        return str(eval(expression, {"__builtins__": {}}, allowed_names))
    except Exception as e:
        return f"Error calculating: {str(e)}"


# Router Agent
router_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="messages"),
    ("system", """You are a routing agent that classifies user queries into one of three categories:
    1. 'code_generator' - Requests for code generation, implementation help, or coding examples
    2. 'researcher' - Questions requiring current information, research, or fact-checking
    3. 'math_solver' - Mathematical problems, calculations, or numerical analysis
    
    Respond ONLY with one of these categories. No other text.""")
])


def router(state: MessagesState) -> Dict[str, Any]:
    """
    Routes incoming queries to the appropriate specialized agent.

    Args:
        state (MessagesState): Current workflow state containing messages

    Returns:
        Dict[str, Any]: Dictionary with 'next_agent' key indicating which agent should handle the query
    """
    messages = state["messages"]
    response = llm.invoke(router_prompt.format_messages(messages=messages))
    category = response.content.strip().lower()
    return {"next_agent": category}


# Code Generator Agent
code_generator_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="messages"),
    ("system", """You are a code generation specialist. Generate clean, well-documented code with:
    - Clear function/variable names
    - Type hints where appropriate
    - Docstrings and comments
    - Error handling""")
])


def code_generator_agent(state: MessagesState) -> Dict[str, Any]:
    messages = state["messages"]
    response = llm.invoke(
        code_generator_prompt.format_messages(messages=messages))
    return {"final_answer": response.content}


# Research Agent with Tavily Search
research_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="messages"),
    ("system", """You are a research specialist with access to web search. 
    Synthesize the search results and provide clear findings with sources.""")
])

tavily_search = TavilySearchResults(max_results=3)


def research_agent(state: MessagesState) -> Dict[str, Any]:
    messages = state["messages"]
    query = messages[-1].content
    search_results = tavily_search.invoke(query)

    research_messages = messages + [
        HumanMessage(
            content=f"Search results: {json.dumps(search_results, indent=2)}")
    ]

    response = llm.invoke(
        research_prompt.format_messages(messages=research_messages))
    return {"final_answer": response.content}


# Math Problem Solver Agent
math_solver_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="messages"),
    ("system", """You are a mathematical problem solver with access to a calculator.
    Break down problems into steps and show your work. You can use the calculate_expression function for calculations.""")
])


def math_solver_agent(state: MessagesState) -> Dict[str, Any]:
    messages = state["messages"]
    response = llm.invoke(
        math_solver_prompt.format_messages(messages=messages))
    return {"final_answer": response.content}

# Define the routing function


def should_continue(state: MessagesState) -> Union[str, Literal[END]]:
    """
    Determines the next step in the workflow based on the current state.

    This function is called after the router to determine where to send the query next.
    If next_agent is set, it routes to that agent. Otherwise, it ends the workflow.

    Args:
        state (MessagesState): Current workflow state

    Returns:
        Union[str, Literal[END]]: Either the name of the next agent or END to finish
    """
    if state.get("next_agent"):
        return state["next_agent"]
    return END


# Create the workflow graph
workflow = StateGraph(MessagesState)

# Graph Construction
# 1. First, add all nodes (agents) to the graph
workflow.add_node("router", router)
workflow.add_node("code_generator", code_generator_agent)
workflow.add_node("researcher", research_agent)
workflow.add_node("math_solver", math_solver_agent)

# 2. Set the entry point (where queries start)
workflow.set_entry_point("router")

# 3. Define the routing logic using conditional edges
# This maps the router's output to the appropriate agent
workflow.add_conditional_edges(
    "router",  # Source node
    should_continue,  # Function that determines the next step
    {
        # Map router outputs to destination nodes
        "code_generator": "code_generator",
        "researcher": "researcher",
        "math_solver": "math_solver"
    }
)

# 4. Set which nodes can end the workflow
workflow.set_finish_point(["code_generator", "researcher", "math_solver"])

# 5. Compile the graph into an executable chain
chain = workflow.compile()

# Example usage with detailed explanation
if __name__ == "__main__":
    """
    Demonstrates the workflow with example queries.

    The workflow follows these steps:
    1. Query is wrapped in a HumanMessage and added to state
    2. Router agent classifies the query type
    3. Query is forwarded to the appropriate specialized agent
    4. Agent processes the query and returns a final answer
    5. Workflow ends and returns the result
    """
    test_queries = [
        "Create a Python function to find the nth Fibonacci number using dynamic programming",
        "What are the latest developments in quantum computing?",
        "Calculate the compound interest on $1000 invested for 5 years at 8% annual interest rate",
    ]

    for query in test_queries:
        print(f"\n{'='*50}\nInput: {query}\n{'='*50}")
        # Initialize state with the query
        result = chain.invoke({
            "messages": [HumanMessage(content=query)],
            "next_agent": None,
            "final_answer": None,
            "context": {}
        })
        print(f"\nRouted to: {result['next_agent']}")
        print(f"\nResponse:\n{result['final_answer']}")
