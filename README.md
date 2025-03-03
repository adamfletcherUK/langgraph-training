# LangGraph Agent Patterns Demo

This repository demonstrates various agent patterns and workflows using LangGraph, showcasing how to build sophisticated AI agent systems. Each pattern demonstrates different aspects of agent design and interaction.

## 🧠 Agent Patterns Overview

### 1. Routing Workflow Pattern
**File**: `routing_workflow.py`
- **Purpose**: Intelligent query routing to specialized agents
- **Key Features**:
  - Router agent that classifies queries
  - Specialized agents for different tasks (code generation, research, math)
  - Conditional routing based on query type
  - State management between agents
- **Key Concepts Demonstrated**:
  - **Conditional Routing**: Uses a router agent to classify queries and direct them to specialized agents
  - **State Management**: Maintains conversation state and context between agent transitions
  - **Graph Construction**: Builds a dynamic graph where the router determines the next agent
  - **Node Types**: Implements both router and specialized processing nodes

### 2. Gated Agent Chain Pattern
**File**: `gated_agent_chain.py`
- **Purpose**: Workflow with decision points and conditional paths
- **Key Features**:
  - Gate/decision point for workflow control
  - Sentiment-based routing
  - Multiple processing paths
  - Early termination capability
- **Key Concepts Demonstrated**:
  - **Decision Points**: Uses a gate node to analyze sentiment and control workflow
  - **Early Termination**: Implements conditional edges to end processing when needed
  - **Linear Flow**: Creates a structured path with potential early exits
  - **State Transitions**: Manages state through a series of transformations

### 3. Recursive Agent Pattern
**File**: `recursive_agent_demo.py`
- **Purpose**: Complex task decomposition and parallel processing
- **Key Features**:
  - Task decomposition into subtasks
  - Parallel execution of subtasks
  - Result aggregation
  - Rate limiting and concurrency control
- **Key Concepts Demonstrated**:
  - **Task Decomposition**: Breaks complex tasks into manageable subtasks
  - **Parallel Processing**: Executes subtasks concurrently with rate limiting
  - **State Aggregation**: Combines results from multiple subtasks
  - **Error Handling**: Implements rate limiting and concurrency control

### 4. Parallel Document Analysis Pattern
**File**: `parallel_document_analysis.py`
- **Purpose**: Concurrent analysis of documents
- **Key Features**:
  - Multiple specialized analysis agents
  - Parallel processing of different aspects
  - Result compilation
  - Web document processing
- **Key Concepts Demonstrated**:
  - **Concurrent Processing**: Multiple analysis agents working in parallel
  - **Multiple Entry Points**: Graph structure allowing parallel execution
  - **Result Compilation**: Aggregation of results from different analysis types
  - **Error Recovery**: Handles failed document fetches gracefully

### 5. Memory-Augmented Workflow Pattern
**File**: `memory_augmented_workflow.py`
- **Purpose**: Long-term memory and context management
- **Key Features**:
  - Vectorized memory storage
  - Semantic search for relevant memories
  - Temporal relationship tracking
  - Knowledge base maintenance
- **Key Concepts Demonstrated**:
  - **Long-term Memory**: Vectorized storage for persistent knowledge
  - **Semantic Search**: Retrieves relevant memories based on context
  - **State Enrichment**: Enhances state with retrieved memories
  - **Temporal Tracking**: Maintains memory timestamps and relationships

### 6. Feedback Loop Pattern
**File**: `feedback_loop.py`
- **Purpose**: Iterative content improvement
- **Key Features**:
  - Content creation and review cycle
  - Quality threshold checking
  - Multiple agent collaboration
  - Iteration tracking
- **Key Concepts Demonstrated**:
  - **Iterative Processing**: Cycles through content creation and review
  - **Quality Gates**: Implements quality thresholds for content
  - **State Tracking**: Maintains iteration count and quality metrics
  - **Conditional Loops**: Routes back to creation if quality not met

### 7. Human-in-the-Loop Pattern
**File**: `hitl_demo.py`
- **Purpose**: Human-AI collaboration
- **Key Features**:
  - Human approval checkpoints
  - Research planning and execution
  - Custom approval callbacks
  - Structured workflow steps
- **Key Concepts Demonstrated**:
  - **Human Interaction**: Integrates human approval checkpoints
  - **Workflow Control**: Manages state transitions based on human decisions
  - **Custom Callbacks**: Allows flexible approval handling
  - **Linear Flow**: Structured path with human decision points

### 8. Chatbot with Tools Pattern
**File**: `chatbot_tools.py`
- **Purpose**: Tool-enabled conversational agent
- **Key Features**:
  - Dynamic tool routing
  - Tool execution management
  - Conversation state tracking
  - Flexible tool integration
- **Key Concepts Demonstrated**:
  - **Dynamic Tool Routing**: Routes to appropriate tools based on needs
  - **State Preservation**: Maintains conversation context
  - **Tool Integration**: Flexible tool registration and execution
  - **Error Recovery**: Handles tool execution failures gracefully

## 🚀 Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## 🎯 Usage Examples

Each pattern can be run independently. Here's a quick example of using the routing workflow:

```python
from routing_workflow import chain

# Initialize state with your query
result = chain.invoke({
    "messages": [HumanMessage(content="Your query here")],
    "next_agent": None,
    "final_answer": None,
    "context": {}
})
```

## 🧠 Learning Resources

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Agent Design Patterns](https://python.langchain.com/docs/modules/agents/agent_types/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details. 