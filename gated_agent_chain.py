"""
LangGraph Tutorial: Building a Gated Agent Workflow
================================================

This tutorial demonstrates how to build a multi-agent workflow with a gate/decision point
using LangGraph. The workflow consists of multiple agents that process text input and
a gate that decides whether to continue or terminate the workflow based on sentiment.

Workflow Structure:
-----------------
[START] -> [First Agent] -> [Gate] -> (if positive) -> [Second Agent] -> [Final Agent] -> [END]
                                  -> (if negative) -> [END]

Key Concepts Demonstrated:
-----------------------
1. State Management: Using TypedDict for maintaining workflow state
2. Agent Implementation: Creating agents with specific responsibilities
3. Conditional Routing: Using gates to make workflow decisions
4. Graph Construction: Building a workflow with START and END nodes

Author: Your Name
Date: March 2024
"""

from typing import Annotated, Sequence, TypedDict, Union
from typing_extensions import TypeVar
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import Graph, MessageGraph, END, START
import json

# Load environment variables for API keys
load_dotenv()

# Initialize the Anthropic model
model = ChatAnthropic(model_name="claude-3-sonnet-20240229")

# Define our state type


class AgentState(TypedDict):
    """
    Represents the state that flows through the graph.

    Attributes:
        messages (Sequence[BaseMessage]): History of messages in the conversation
        next_step (str): Indicates the next node to execute in the workflow
    """
    messages: Sequence[BaseMessage]
    next_step: str


def first_agent(state: AgentState) -> AgentState:
    """
    First agent that processes the initial input.

    This agent takes the user's input and creates a brief summary.
    It acts as an initial filter and contextualizer for the workflow.

    Args:
        state (AgentState): Current workflow state with messages

    Returns:
        AgentState: Updated state with summary and next step
    """
    messages = state["messages"]
    response = model.invoke(
        messages +
        [HumanMessage(
            content="Analyze this request and provide a brief summary. Keep it under 50 words.")]
    )
    return {
        "messages": messages + [response],
        "next_step": "gate"
    }


def gate_node(state: AgentState) -> dict:
    """
    Gate that decides whether to continue or exit the workflow.

    This node acts as a decision point, analyzing the sentiment of the previous
    message to determine whether to proceed with deeper analysis or terminate.

    Args:
        state (AgentState): Current workflow state

    Returns:
        dict: Updated state with decision path ("pass" or "fail")
    """
    messages = state["messages"]
    last_message = messages[-1].content

    # Simple gate logic - check if the message is positive
    response = model.invoke(
        messages +
        [HumanMessage(
            content="Is this message generally positive? Answer with only 'yes' or 'no'.")]
    )

    is_positive = "yes" in response.content.lower()

    # Return the state and edge to follow
    return {
        "messages": state["messages"] + [response],
        "next_step": "pass" if is_positive else "fail"
    }


def second_agent(state: AgentState) -> AgentState:
    """
    Second agent that processes passed requests.

    This agent only runs for positive sentiment messages and provides
    a more detailed analysis of the input.

    Args:
        state (AgentState): Current workflow state

    Returns:
        AgentState: Updated state with detailed analysis
    """
    messages = state["messages"]
    response = model.invoke(
        messages +
        [HumanMessage(
            content="Expand on the previous analysis with more details.")]
    )
    return {
        "messages": messages + [response],
        "next_step": "final_agent"
    }


def final_agent(state: AgentState) -> AgentState:
    """
    Final agent that provides the conclusion.

    This agent synthesizes all previous analyses into a final conclusion.
    Only reached for positive sentiment paths.

    Args:
        state (AgentState): Current workflow state

    Returns:
        AgentState: Final state with conclusion
    """
    messages = state["messages"]
    response = model.invoke(
        messages +
        [HumanMessage(
            content="Provide a final conclusion based on all previous analyses.")]
    )
    return {
        "messages": messages + [response],
        "next_step": "end"
    }


# Create the graph
workflow = Graph()

# Add nodes to the graph
workflow.add_node("first_agent", first_agent)
workflow.add_node("gate", gate_node)
workflow.add_node("second_agent", second_agent)
workflow.add_node("final_agent", final_agent)

# Build the graph structure
# 1. Connect START to first_agent (entry point)
workflow.add_edge(START, "first_agent")

# 2. Connect first_agent to gate (always proceeds to decision point)
workflow.add_edge("first_agent", "gate")

# 3. Set up conditional routing from gate
workflow.add_conditional_edges(
    "gate",
    lambda x: x["next_step"],  # Function to extract routing decision
    {
        "pass": "second_agent",  # Positive sentiment path
        "fail": END  # Negative sentiment path - terminate
    }
)

# 4. Connect remaining nodes in positive path
workflow.add_edge("second_agent", "final_agent")
workflow.add_edge("final_agent", END)

# Compile the graph
app = workflow.compile()


def run_workflow(input_text: str):
    """
    Executes the workflow with the given input text.

    This function initializes the workflow state and processes the input
    through the agent graph, collecting responses at each step.

    Args:
        input_text (str): The text to analyze

    Returns:
        list: List of message contents from the workflow execution
    """
    result = app.invoke({
        "messages": [HumanMessage(content=input_text)],
        "next_step": "first_agent"
    })

    # Extract messages for better readability
    messages = [msg.content for msg in result["messages"]]
    return messages


if __name__ == "__main__":
    """
    Demo execution of the workflow with positive and negative examples.

    This demonstrates how the workflow handles different inputs:
    1. Positive input: Goes through the entire chain
    2. Negative input: Terminates after the gate
    """
    # Test with a positive input
    print("\n=== Testing with positive input ===")
    positive_result = run_workflow(
        "I'm excited to learn about AI and its potential to help humanity!")
    print("\nWorkflow results:")
    for idx, msg in enumerate(positive_result[1:], 1):
        print(f"\nStep {idx}:\n{msg}")

    # Test with a negative input
    print("\n\n=== Testing with negative input ===")
    negative_result = run_workflow(
        "I'm worried about the negative impacts of technology.")
    print("\nWorkflow results:")
    for idx, msg in enumerate(negative_result[1:], 1):
        print(f"\nStep {idx}:\n{msg}")
