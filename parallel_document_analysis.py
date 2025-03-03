from typing import Dict, TypedDict, Annotated, Sequence
import operator
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
import requests
from bs4 import BeautifulSoup
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define our state structure


class AgentState(TypedDict):
    tone_analysis: str
    clarity_analysis: str
    environmental_analysis: str
    final_report: str
    document_text: str


def fetch_document(url: str) -> str:
    """Fetch and parse the document from the given URL."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Extract main content - adjust selectors based on the website structure
    content = soup.get_text()
    return content

# Define our analysis agents


def create_tone_agent():
    llm = ChatAnthropic(
        model="claude-3-sonnet-20240229",
        temperature=0,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    def analyze_tone(state: AgentState) -> Dict:
        response = llm.invoke(
            [HumanMessage(content=f"""
            Analyze the tone of voice in this text. Consider factors like:
            - Formality level
            - Professional vs casual language
            - Authority and confidence
            - Emotional undertones
            
            Text: {state['document_text']}
            
            Provide a concise analysis in 3-4 sentences.
            """)]
        )
        return {"tone_analysis": response.content}

    return analyze_tone


def create_clarity_agent():
    llm = ChatAnthropic(
        model="claude-3-sonnet-20240229",
        temperature=0,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    def analyze_clarity(state: AgentState) -> Dict:
        response = llm.invoke(
            [HumanMessage(content=f"""
            Analyze the clarity and readability of this text. Consider:
            - Sentence structure complexity
            - Use of technical terms
            - Information organization
            - Accessibility to general audience
            
            Text: {state['document_text']}
            
            Provide a concise analysis in 3-4 sentences.
            """)]
        )
        return {"clarity_analysis": response.content}

    return analyze_clarity


def create_environmental_agent():
    llm = ChatAnthropic(
        model="claude-3-sonnet-20240229",
        temperature=0,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    def analyze_environmental_impact(state: AgentState) -> Dict:
        response = llm.invoke(
            [HumanMessage(content=f"""
            Analyze the environmental considerations in this text. Look for:
            - Environmental impact mentions
            - Sustainability practices
            - Conservation measures
            - Climate considerations
            
            Text: {state['document_text']}
            
            Provide a concise analysis in 3-4 sentences.
            """)]
        )
        return {"environmental_analysis": response.content}

    return analyze_environmental_impact


def create_report_compiler():
    llm = ChatAnthropic(
        model="claude-3-sonnet-20240229",
        temperature=0,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    def compile_report(state: AgentState) -> Dict:
        response = llm.invoke(
            [HumanMessage(content=f"""
            Create a comprehensive report combining the following analyses:
            
            Tone Analysis:
            {state['tone_analysis']}
            
            Clarity Analysis:
            {state['clarity_analysis']}
            
            Environmental Impact Analysis:
            {state['environmental_analysis']}
            
            Synthesize these analyses into a cohesive summary report highlighting key findings.
            """)]
        )
        return {"final_report": response.content}

    return compile_report


def create_workflow(url: str):
    # Initialize the graph
    workflow = StateGraph(AgentState)

    # Add nodes for each agent - with '_node' suffix to differentiate from state keys
    workflow.add_node("tone_node", create_tone_agent())
    workflow.add_node("clarity_node", create_clarity_agent())
    workflow.add_node("environmental_node", create_environmental_agent())
    workflow.add_node("compiler_node", create_report_compiler())

    # Add parallel edges - updated with new node names
    workflow.add_edge("tone_node", "compiler_node")
    workflow.add_edge("clarity_node", "compiler_node")
    workflow.add_edge("environmental_node", "compiler_node")

    # Set the final node to end
    workflow.add_edge("compiler_node", END)

    # Set entry points - updated with new node names
    workflow.set_entry_point("tone_node")
    workflow.set_entry_point("clarity_node")
    workflow.set_entry_point("environmental_node")

    # Compile the graph
    app = workflow.compile()

    # Create initial state
    document_text = fetch_document(url)
    initial_state = {
        "document_text": document_text,
        "tone_analysis": "",
        "clarity_analysis": "",
        "environmental_analysis": "",
        "final_report": ""
    }

    return app, initial_state


def main():
    url = "https://www.gov.uk/find-funding-for-land-or-farms/csam1-assess-soil-produce-a-soil-management-plan-and-test-soil-organic-matter"
    app, initial_state = create_workflow(url)

    # Run the workflow
    result = app.invoke(initial_state)

    # Print the final report
    print("\nFinal Analysis Report:")
    print("=" * 50)
    print(result["final_report"])


if __name__ == "__main__":
    main()
