"""
Recursive Agent Pattern Demo using LangGraph

This script demonstrates how to implement a recursive agent pattern where:
1. A main agent breaks down complex tasks into subtasks
2. Specialized agents handle specific subtasks
3. Results are combined recursively
4. Task dependencies are managed

Key Concepts:
- Task Decomposition
- Recursive Execution
- Result Aggregation
- State Management
"""

from typing import Annotated, Dict, List, TypedDict, Union, Literal
from typing_extensions import TypeVar
from langgraph.graph import StateGraph, END, START
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatAnthropic(model="claude-3-sonnet-20240229")

# Define our state type


class TaskState(TypedDict):
    """
    Represents the state of a task in the recursive workflow.

    Attributes:
        messages: List of conversation messages
        task: Current task description
        subtasks: List of decomposed subtasks
        results: Dictionary of task results
        depth: Current recursion depth
        parent_task: Reference to parent task if any
        final_result: The final aggregated result
    """
    messages: List[BaseMessage]
    task: str
    subtasks: List[str]
    results: Dict[str, str]
    depth: int
    parent_task: str | None
    final_result: str | None


# Define prompts for different agent roles
task_decomposer_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="messages"),
    ("system", """You are a task decomposition specialist. Your role is to:
    1. Analyze complex tasks
    2. Break them down into smaller, manageable subtasks
    3. Identify dependencies between subtasks
    4. Ensure each subtask is specific and actionable
    
    For each task, you MUST provide a JSON response with the following structure:
    {{
        "subtasks": [
            "First specific subtask",
            "Second specific subtask",
            "Third specific subtask",
            ...
        ],
        "dependencies": {{
            "subtask_name": ["depends_on_subtask1", "depends_on_subtask2"]
        }},
        "complexity": {{
            "subtask_name": "high|medium|low"
        }}
    }}
    
    Guidelines:
    1. Each subtask should be a single, specific action
    2. Break down complex tasks into 3-5 subtasks
    3. Make subtasks concrete and actionable
    4. Include clear dependencies if any
    5. Rate complexity for each subtask
    
    Example for "Create a marketing strategy":
    {{
        "subtasks": [
            "Conduct market research on eco-friendly products",
            "Analyze competitor pricing and positioning",
            "Identify target audience demographics and preferences",
            "Develop pricing strategy based on market data",
            "Create promotional plan with specific channels"
        ],
        "dependencies": {{
            "Develop pricing strategy based on market data": ["Conduct market research on eco-friendly products", "Analyze competitor pricing and positioning"],
            "Create promotional plan with specific channels": ["Identify target audience demographics and preferences"]
        }},
        "complexity": {{
            "Conduct market research on eco-friendly products": "high",
            "Analyze competitor pricing and positioning": "medium",
            "Identify target audience demographics and preferences": "medium",
            "Develop pricing strategy based on market data": "high",
            "Create promotional plan with specific channels": "medium"
        }}
    }}
    """)
])

specialist_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="messages"),
    ("system", """You are a specialized task executor. Your role is to:
    1. Execute specific subtasks
    2. Provide detailed results
    3. Report any issues or blockers
    
    Be thorough and precise in your execution.
    """)
])

result_aggregator_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="messages"),
    ("system", """You are a result aggregation specialist. Your role is to:
    1. Combine results from multiple subtasks
    2. Ensure consistency across results
    3. Handle any conflicts or discrepancies
    4. Provide a cohesive final output
    
    Synthesize the results into a comprehensive solution.
    """)
])


def decompose_task(state: TaskState) -> Dict:
    """
    Decomposes a complex task into subtasks.
    """
    logger.info(f"🔍 Starting task decomposition for: {state['task'][:100]}...")
    messages = state["messages"]
    response = llm.invoke(
        task_decomposer_prompt.format_messages(
            messages=messages +
            [HumanMessage(content=f"Decompose this task: {state['task']}")]
        )
    )

    # Parse the response to get subtasks
    try:
        result = json.loads(response.content)
        subtasks = result.get("subtasks", [])
        dependencies = result.get("dependencies", {})
        complexity = result.get("complexity", {})

        if not subtasks:
            logger.warning("No subtasks found in response, using fallback")
            subtasks = [response.content]
        else:
            logger.info(f"📋 Created {len(subtasks)} subtasks:")
            for i, task in enumerate(subtasks, 1):
                deps = dependencies.get(task, [])
                comp = complexity.get(task, "unknown")
                logger.info(f"  {i}. {task[:100]}... (Complexity: {comp})")
                if deps:
                    logger.info(f"     Dependencies: {', '.join(deps)}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        logger.error(f"Raw response: {response.content}")
        subtasks = [response.content]
    except Exception as e:
        logger.error(f"Error parsing subtasks: {e}")
        subtasks = [response.content]

    return {
        "subtasks": subtasks,
        "messages": messages + [response],
        "final_result": None
    }


def execute_subtask(state: TaskState) -> Dict:
    """
    Executes a specific subtask.
    """
    current_task = state["subtasks"][0]
    logger.info(f"⚙️  Executing subtask: {current_task[:100]}...")

    messages = state["messages"]
    remaining_subtasks = state["subtasks"][1:]

    response = llm.invoke(
        specialist_prompt.format_messages(
            messages=messages +
            [HumanMessage(content=f"Execute this subtask: {current_task}")]
        )
    )

    # Update results with the current task's result
    current_results = state["results"].copy()
    current_results[current_task] = response.content

    logger.info(f"✅ Completed subtask: {current_task[:100]}...")
    logger.info(f"📊 Remaining subtasks: {len(remaining_subtasks)}")

    return {
        "subtasks": remaining_subtasks,
        "results": current_results,
        "messages": messages + [response],
        "final_result": None
    }


def aggregate_results(state: TaskState) -> Dict:
    """
    Aggregates results from multiple subtasks.
    """
    logger.info("🔄 Starting results aggregation...")
    logger.info(f"📊 Aggregating {len(state['results'])} subtask results")

    messages = state["messages"]
    results = state["results"]

    response = llm.invoke(
        result_aggregator_prompt.format_messages(
            messages=messages +
            [HumanMessage(content=f"Aggregate these results: {results}")]
        )
    )

    logger.info("✨ Results aggregation completed")
    return {
        "final_result": response.content,
        "messages": messages + [response]
    }


def should_continue(state: TaskState) -> Union[str, Literal[END]]:
    """
    Determines whether to continue with subtasks or end the workflow.
    """
    if len(state["subtasks"]) > 0:
        logger.info(
            f"🔄 Continuing with {len(state['subtasks'])} remaining subtasks")
        return "execute_subtask"
    elif state["parent_task"] is None:
        logger.info("📝 All subtasks completed, proceeding to aggregation")
        return "aggregate_results"
    else:
        logger.info("🏁 Workflow completed")
        return END


def create_recursive_workflow():
    """
    Creates a recursive agent workflow.
    """
    logger.info("🚀 Initializing recursive workflow")
    # Initialize the graph
    workflow = StateGraph(TaskState)

    # Add nodes
    workflow.add_node("decompose", decompose_task)
    workflow.add_node("execute_subtask", execute_subtask)
    workflow.add_node("aggregate_results", aggregate_results)

    # Add edges
    workflow.add_edge(START, "decompose")
    workflow.add_edge("decompose", "execute_subtask")
    workflow.add_conditional_edges(
        "execute_subtask",
        should_continue,
        {
            "execute_subtask": "execute_subtask",
            "aggregate_results": "aggregate_results",
            END: END
        }
    )
    workflow.add_edge("aggregate_results", END)

    # Compile the graph
    return workflow.compile()


def main():
    """
    Demonstrates the recursive agent pattern with a complex task.
    """
    logger.info("🎬 Starting recursive agent demo")

    # Create the workflow
    app = create_recursive_workflow()

    # Example complex task
    complex_task = """
    Create a comprehensive marketing strategy for a new eco-friendly product line.
    Include market research, competitor analysis, target audience identification,
    pricing strategy, and promotional plan.
    """

    logger.info(f"📝 Initial task: {complex_task[:100]}...")

    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=complex_task)],
        "task": complex_task,
        "subtasks": [],
        "results": {},
        "depth": 0,
        "parent_task": None,
        "final_result": None
    }

    # Run the workflow
    logger.info("⚙️  Starting workflow execution")
    start_time = datetime.now()
    result = app.invoke(initial_state)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info(f"⏱️  Workflow completed in {duration:.2f} seconds")

    # Print the final result
    print("\nFinal Marketing Strategy:")
    print("=" * 50)
    print(result["final_result"])


if __name__ == "__main__":
    main()
