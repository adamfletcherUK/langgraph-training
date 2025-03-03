"""
LangGraph Feedback Loop Implementation

This module implements a feedback loop system using LangGraph where:
1. Agent A creates content
2. Agent B reviews and provides feedback
3. The loop continues until quality threshold is met
4. Agent C processes the final approved content

Author: [Your Name]
Date: [Current Date]
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")


class State(TypedDict):
    """
    Represents the conversation state with feedback loop tracking.

    Attributes:
        messages: List of conversation messages
        feedback: Current feedback from Agent B
        status: Current status (PASS/FAIL/PENDING)
        iterations: Number of feedback iterations completed
    """
    messages: Annotated[list, add_messages]
    feedback: str
    status: str
    iterations: int


def agent_a(state: State):
    """
    Creator agent that generates initial content.
    Takes into account previous feedback if available.
    """
    # Start with the initial user request
    base_messages = [state["messages"][0]]  # Keep the original user request

    # If there's feedback, add it as a system message at the beginning
    if state.get("feedback"):
        base_messages.insert(0, {
            "role": "system",
            "content": f"Previous feedback: {state['feedback']}. Please improve the content based on this feedback."
        })

    response = llm.invoke(base_messages)
    return {"messages": [response]}


def agent_b(state: State):
    """
    QA agent that reviews content and provides structured feedback.
    Returns a PASS/FAIL decision and detailed feedback.
    """
    messages = state["messages"]
    last_content = messages[-1].content

    prompt = f"""Review this content and decide if it meets high quality standards.
    
    Provide your response in the following format:
    Decision: [PASS or FAIL]
    Feedback: [your detailed feedback explaining why it passed or failed and any suggested improvements]
    
    Consider these aspects in your evaluation:
    1. Technical accuracy
    2. Clarity and structure
    3. Completeness
    4. Engagement
    
    Content to review:
    {last_content}
    """

    review = llm.invoke([{"role": "user", "content": prompt}])

    # Parse review to extract decision and feedback
    try:
        decision_line = [line for line in review.content.split(
            '\n') if 'Decision:' in line][0]
        status = decision_line.split('Decision:')[1].strip()
    except:
        status = "FAIL"  # Default to fail if parsing fails

    return {
        "feedback": review.content,
        "status": status,
        "messages": [review],
        "iterations": state["iterations"] + 1
    }


def agent_c(state: State):
    """
    Final processor that handles approved content.
    Only receives content that has passed quality review.
    """
    messages = state["messages"]
    prompt = """The following content has passed our quality review process. 
    Please format it for final publication, ensuring it maintains all the quality aspects
    while optimizing for readability and engagement.
    """
    messages.append({"role": "system", "content": prompt})
    return {"messages": [llm.invoke(messages)]}


def route_feedback(state: State):
    """
    Determines whether to continue the feedback loop or move to final processing.

    Rules:
    - If status is PASS, move to final processing
    - If max iterations (3) reached, move to final processing
    - Otherwise, continue feedback loop
    """
    if state["status"] == "PASS":
        print(f"Content passed quality review!")
        return "agent_c"
    elif state["iterations"] >= 3:
        print("Maximum iterations reached")
        return "agent_c"
    else:
        print(f"Content needs improvement (Iteration {state['iterations']})")
        return "agent_a"


def build_graph():
    """Constructs the feedback loop graph."""
    graph_builder = StateGraph(State)

    # Add nodes
    graph_builder.add_node("agent_a", agent_a)
    graph_builder.add_node("agent_b", agent_b)
    graph_builder.add_node("agent_c", agent_c)

    # Add edges
    graph_builder.add_edge(START, "agent_a")
    graph_builder.add_edge("agent_a", "agent_b")

    # Add conditional edge from agent_b
    graph_builder.add_conditional_edges(
        "agent_b",
        route_feedback,
        {
            "agent_a": "agent_a",  # Continue feedback loop
            "agent_c": "agent_c"   # Move to final processing
        }
    )

    graph_builder.add_edge("agent_c", END)

    return graph_builder.compile()


def run_feedback_loop(initial_input: str):
    """
    Runs the feedback loop with the given input.

    Args:
        initial_input: The initial content or request to process
    """
    graph = build_graph()

    initial_state = {
        "messages": [{"role": "user", "content": initial_input}],
        "feedback": "",
        "status": "PENDING",
        "iterations": 0
    }

    print("Starting feedback loop...")
    print("-" * 50)

    for event in graph.stream(initial_state):
        for value in event.values():
            if "feedback" in value:
                print(f"\nReview (Status: {value['status']}):")
                print(value['feedback'])
                print("-" * 50)
            else:
                print("\nOutput:")
                print(value['messages'][-1].content)
                print("-" * 50)


def main():
    """Main execution function."""
    # Example usage
    test_input = "Please write hastely written poem about bears - just output the poem only!"
    run_feedback_loop(test_input)


if __name__ == "__main__":
    main()
