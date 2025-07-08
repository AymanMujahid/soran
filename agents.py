# agents.py
from langchain.agents import Tool, AgentExecutor, initialize_agent
from langchain.llms import OpenAI
from report_generator import generate_report
from alert_agent import check_alerts
from scheduler_agent import schedule_task
from vector_store import query_embeddings

tools = [
    Tool(
        name="ReportGenerator",
        func=generate_report,
        description="Generate periodic summary reports from the knowledge base"
    ),
    Tool(
        name="AlertAgent",
        func=check_alerts,
        description="Monitor metrics and send alerts when thresholds are crossed"
    ),
    Tool(
        name="SchedulerAgent",
        func=schedule_task,
        description="Schedule reminders or tasks based on user input"
    ),
    Tool(
        name="KnowledgeRetriever",
        func=lambda query, bot_id: query_embeddings(bot_id, top_k=5, query_emb=query),
        description="Retrieve relevant document chunks for a given query"
    )
]

def create_agent():
    llm = OpenAI(model_name="mistral-7b", temperature=0)
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=False)
    return agent

def run_agent(agent, tool_name: str, *args, **kwargs):
    return agent.run(f"{tool_name}: {args} {kwargs}")
