from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

web_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    description="Web Research Agent",
    tools=[DuckDuckGoTools()],
    instructions="Always cite sources",
    markdown=True,
    show_tool_calls=True,
)

finance_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    description="Finance Agent",
    tools=[YFinanceTools(stock_price=True, company_info=True)],
    instructions="Summarize financial data with tables",
    markdown=True,
    show_tool_calls=True,
)

youtube_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    description="YouTube Data Agent",
    tools=[DuckDuckGoTools()],  # Replace with YouTube tool if available
    instructions="Gather and summarize YouTube video data for queries.",
    markdown=True,
    show_tool_calls=True,
)

articles_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    description="Articles Summarizer Agent",
    tools=[DuckDuckGoTools()],
    instructions="Find articles and summarize them for LinkedIn posts.",
    markdown=True,
    show_tool_calls=True,
)

linkedin_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    description="LinkedIn Post Generator Agent",
    tools=[DuckDuckGoTools()],
    instructions="Combine research and generate LinkedIn-ready summaries.",
    markdown=True,
    show_tool_calls=True,
)
