"""
Human-in-the-Loop (HITL) Workflow Demo with LangGraph
===================================================

This script demonstrates how to implement a Human-in-the-Loop workflow using LangGraph.
It creates a research assistant that collaborates with humans during the research process.

Key Concepts:
------------
1. State Management: Using TypedDict to maintain workflow state
2. Workflow Steps: Defined using Enum for clear state transitions
3. Graph Structure: Nodes and edges defining the workflow
4. Human Interaction: Approval checkpoints for human oversight
5. Error Handling: JSON validation and graceful error recovery

Usage:
------
Basic usage:
    result = run_research_workflow("What are the major AI breakthroughs in 2024?")

With custom approval callback:
    def my_approval(plan): 
        # Custom approval logic
        return True
    
    result = run_research_workflow(
        "What are the major AI breakthroughs in 2024?",
        approval_callback=my_approval
    )
"""

from typing import Annotated, Any, Dict, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import Graph, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_anthropic import ChatAnthropic
import json
from enum import Enum

# === State Definition ===


class State(TypedDict):
    """
    Represents the current state of the workflow.

    Attributes:
        messages (list[BaseMessage]): Conversation history
        next_step (str): Next step in the workflow
        research_plan (Dict[str, Any]): Structured research plan
        final_result (str): Final research findings
        human_approved (bool): Whether the plan was approved
    """
    messages: list[BaseMessage]
    next_step: str
    research_plan: Dict[str, Any]
    final_result: str
    human_approved: bool

# === Workflow Steps ===


class WorkflowStep(str, Enum):
    """
    Defines the possible steps in the workflow.

    Steps:
        PLAN: Initial research plan creation
        WAIT_FOR_APPROVAL: Human review checkpoint
        EXECUTE: Conduct approved research
        PRESENT: Format and present findings
        END: Workflow completion
    """
    PLAN = "plan"
    WAIT_FOR_APPROVAL = "wait_for_approval"
    EXECUTE = "execute"
    PRESENT = "present"
    END = "end"


# === Model Initialization ===
model = ChatAnthropic(model="claude-3-sonnet-20240229")

# === Prompt Templates ===
# Each prompt template defines the interaction with the AI model at different stages
plan_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="messages"),
    ("system", """You are a research assistant. Create a detailed plan to answer the user's research question.
    Format your response as a JSON string with the following structure:
    {{
        "steps": ["list", "of", "research", "steps"],
        "rationale": "explanation of your approach",
        "estimated_time": "time estimate"
    }}
    """)
])

execute_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Execute the approved research plan and compile the findings.")
])

present_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Present the research findings in a clear, concise format.")
])

# === Workflow Nodes ===


def create_plan(state: State) -> State:
    """
    Creates a research plan based on the user's question.

    This node:
    1. Takes the current conversation state
    2. Generates a structured research plan using the AI model
    3. Updates the state with the plan and requests human approval

    Error Handling:
    - Validates JSON format of the response
    - Gracefully handles parsing errors
    """
    messages = state["messages"]
    response = model.invoke(plan_prompt.invoke(
        {"messages": messages}).to_messages())

    try:
        plan = json.loads(response.content)
        state["research_plan"] = plan
        state["next_step"] = WorkflowStep.WAIT_FOR_APPROVAL
        state["messages"].append(AIMessage(
            content=f"Proposed Research Plan:\n{json.dumps(plan, indent=2)}\n\nDo you approve this plan? (yes/no)"))
    except json.JSONDecodeError:
        state["messages"].append(
            AIMessage(content="Error: Could not create research plan. Please try again."))
        state["next_step"] = WorkflowStep.END

    return state


def wait_for_approval(state: State) -> State:
    """
    Human-in-the-Loop checkpoint for plan approval.

    This node:
    1. Checks the last message for human approval
    2. Updates workflow state based on the decision
    3. Determines next step (EXECUTE or END)
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        approval = last_message.content.strip().lower()
        state["human_approved"] = approval == "yes"
        state["next_step"] = WorkflowStep.EXECUTE if state["human_approved"] else WorkflowStep.END
    return state


def execute_research(state: State) -> State:
    """
    Executes the approved research plan.

    This node:
    1. Adds the approved plan to the context
    2. Conducts research using the AI model
    3. Stores results and prepares for presentation
    """
    messages = state["messages"]
    plan = state["research_plan"]

    messages.append(
        AIMessage(content=f"Executing approved plan: {json.dumps(plan, indent=2)}"))

    response = model.invoke(execute_prompt.invoke(
        {"messages": messages}).to_messages())
    state["messages"].append(response)
    state["final_result"] = response.content
    state["next_step"] = WorkflowStep.PRESENT

    return state


def present_findings(state: State) -> State:
    """
    Formats and presents the research findings.

    This node:
    1. Takes the research results
    2. Formats them for presentation
    3. Prepares for workflow completion
    """
    messages = state["messages"]
    response = model.invoke(present_prompt.invoke(
        {"messages": messages}).to_messages())
    state["messages"].append(response)
    state["next_step"] = WorkflowStep.END
    return state


# === Workflow Graph Construction ===
# Create the graph structure that defines how the workflow progresses
workflow = StateGraph(State)

# Add nodes to the graph
workflow.add_node("plan", create_plan)
workflow.add_node("wait_for_approval", wait_for_approval)
workflow.add_node("execute", execute_research)
workflow.add_node("present", present_findings)
workflow.add_node("end", lambda x: x)  # Terminal node

# Define the flow between nodes
workflow.add_edge("plan", "wait_for_approval")
workflow.add_conditional_edges(
    "wait_for_approval",
    lambda x: x["next_step"]
)
workflow.add_edge("execute", "present")
workflow.add_edge("present", "end")

# Set the entry point
workflow.set_entry_point("plan")

# Compile the graph
app = workflow.compile()


def run_research_workflow(
    research_question: str,
    approval_callback: callable = None
) -> Dict[str, Any]:
    """
    Main entry point for running the research workflow.

    This function:
    1. Initializes the workflow state
    2. Handles the approval process (via callback or console)
    3. Executes the complete workflow
    4. Returns structured results

    Args:
        research_question (str): The research question to investigate
        approval_callback (callable, optional): Custom function for handling approvals
            Must take a plan (str) and return bool

    Returns:
        Dict[str, Any]: Results dictionary containing:
            - success: Whether the workflow completed
            - result: Final research findings
            - plan: The research plan used
            - messages: Complete conversation history
    """
    initial_state = State(
        messages=[HumanMessage(content=research_question)],
        next_step="plan",
        research_plan={},
        final_result="",
        human_approved=False
    )

    # Run the planning phase
    state = app.invoke(initial_state)

    # Get the proposed plan
    plan = state["messages"][-1].content

    # Handle approval through callback or console
    if approval_callback is None:
        print("\nResearch Assistant:", plan)
        approval = input("\nYour response (yes/no): ").strip().lower() == "yes"
    else:
        approval = approval_callback(plan)

    # Update state with approval decision
    state["messages"].append(HumanMessage(content="yes" if approval else "no"))

    # Complete workflow based on approval
    if approval:
        final_state = app.invoke(state)
        return {
            "success": True,
            "result": final_state["final_result"],
            "plan": final_state["research_plan"],
            "messages": final_state["messages"]
        }
    else:
        return {
            "success": False,
            "result": "Research plan rejected",
            "plan": state["research_plan"],
            "messages": state["messages"]
        }


# === Example Usage ===
if __name__ == "__main__":
    # Demonstrate basic usage
    result = run_research_workflow(
        "What are the major AI breakthroughs in 2024?")

    if result["success"]:
        print("\nFinal Results:")
        print(result["result"])
    else:
        print("\nWorkflow terminated:", result["result"])
