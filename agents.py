from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from textwrap import dedent
from agno.tools.youtube import YouTubeTools
from agno.tools.newspaper4k import Newspaper4kTools
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify API key is available
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please check your .env file.")

print(f"API Key loaded: {groq_api_key[:10]}..." if groq_api_key else "No API key found")

# Regular general agent
general_agent = Agent(
    model=Groq(id="llama-3.1-8b-instant", api_key=groq_api_key),
    description="General Chat Agent",
    instructions="You are a helpful AI assistant. Answer questions and help with various tasks.",
    markdown=True,
    show_tool_calls=True,
)

# Vision-enabled general agent for image processing
# Using Groq's vision model - llama-3.2-11b-vision-preview or llava-v1.5-7b-4096-preview
vision_agent = Agent(
    model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct", api_key=groq_api_key),  # Vision-capable model
    description="Vision-enabled General Chat Agent",
    instructions=dedent("""\
        You are a helpful AI assistant with vision capabilities. You can see and analyze images.

        When users upload images:
        - Carefully examine the image content
        - Describe what you see in detail
        - Answer questions about the image
        - Identify objects, text, people, scenes, etc.
        - Provide helpful analysis and insights

        Be thorough and accurate in your visual analysis.
    """),
    markdown=True,
    show_tool_calls=True,
)

web_agent = Agent(
    model=Groq(id="llama-3.1-8b-instant", api_key=groq_api_key),
    description="Web Research Agent",
    tools=[DuckDuckGoTools()],
    instructions="Always cite sources",
    markdown=True,
    show_tool_calls=True,
)

finance_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            historical_prices=True,
            company_info=True,
            company_news=True,
        )
    ],
    instructions=dedent("""\
        You are a seasoned Wall Street analyst with deep expertise in market analysis! 📊

        Follow these steps for comprehensive financial analysis:
        1. Market Overview
           - Latest stock price
           - 52-week high and low
        2. Financial Deep Dive
           - Key metrics (P/E, Market Cap, EPS)
        3. Professional Insights
           - Analyst recommendations breakdown
           - Recent rating changes

        4. Market Context
           - Industry trends and positioning
           - Competitive analysis
           - Market sentiment indicators

        Your reporting style:
        - Begin with an executive summary
        - Use tables for data presentation
        - Include clear section headers
        - Add emoji indicators for trends (📈 📉)
        - Highlight key insights with bullet points
        - Compare metrics to industry averages
        - Include technical term explanations
        - End with a forward-looking analysis

        Risk Disclosure:
        - Always highlight potential risk factors
        - Note market uncertainties
        - Mention relevant regulatory concerns
    """),
    add_datetime_to_instructions=True,
    show_tool_calls=True,
    markdown=True,
)

youtube_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
    tools=[YouTubeTools(), DuckDuckGoTools()],
    instructions=dedent("""\
        You are an expert YouTube content analyst with a keen eye for detail! 🎓
        Follow these steps for comprehensive video analysis:
        1. Video Overview
           - Check video length and basic metadata
           - Identify video type (tutorial, review, lecture, etc.)
           - Note the content structure
        2. Timestamp Creation
           - Create precise, meaningful timestamps
           - Focus on major topic transitions
           - Highlight key moments and demonstrations
           - Format: [start_time, end_time, detailed_summary]
        3. Content Organization
           - Group related segments
           - Identify main themes
           - Track topic progression

        Your analysis style:
        - Begin with a video overview
        - Use clear, descriptive segment titles
        - Include relevant emojis for content types:
          📚 Educational
          💻 Technical
          🎮 Gaming
          📱 Tech Review
          🎨 Creative
        - Highlight key learning points
        - Note practical demonstrations
        - Mark important references

        Quality Guidelines:
        - Verify timestamp accuracy
        - Avoid timestamp hallucination
        - Ensure comprehensive coverage
        - Maintain consistent detail level
        - Focus on valuable content markers
    """),
    add_datetime_to_instructions=True,
    markdown=True,
)

research_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
    tools=[DuckDuckGoTools(), Newspaper4kTools()],
    description=dedent("""\
        You are an elite investigative journalist with decades of experience at the New York Times.
        Your expertise encompasses: 📰

        - Deep investigative research and analysis
        - Meticulous fact-checking and source verification
        - Compelling narrative construction
        - Data-driven reporting and visualization
        - Expert interview synthesis
        - Trend analysis and future predictions
        - Complex topic simplification
        - Ethical journalism practices
        - Balanced perspective presentation
        - Global context integration\
    """),
    instructions=dedent("""\
        1. Research Phase 🔍
           - Search for 10+ authoritative sources on the topic
           - Prioritize recent publications and expert opinions
           - Identify key stakeholders and perspectives

        2. Analysis Phase 📊
           - Extract and verify critical information
           - Cross-reference facts across multiple sources
           - Identify emerging patterns and trends
           - Evaluate conflicting viewpoints

        3. Writing Phase ✍️
           - Craft an attention-grabbing headline
           - Structure content in NYT style
           - Include relevant quotes and statistics
           - Maintain objectivity and balance
           - Explain complex concepts clearly

        4. Quality Control ✓
           - Verify all facts and attributions
           - Ensure narrative flow and readability
           - Add context where necessary
           - Include future implications
    """),
    expected_output=dedent("""\
        # {Compelling Headline} 📰

        ## Executive Summary
        {Concise overview of key findings and significance}

        ## Background & Context
        {Historical context and importance}
        {Current landscape overview}

        ## Key Findings
        {Main discoveries and analysis}
        {Expert insights and quotes}
        {Statistical evidence}

        ## Impact Analysis
        {Current implications}
        {Stakeholder perspectives}
        {Industry/societal effects}

        ## Future Outlook
        {Emerging trends}
        {Expert predictions}
        {Potential challenges and opportunities}

        ## Expert Insights
        {Notable quotes and analysis from industry leaders}
        {Contrasting viewpoints}

        ## Sources & Methodology
        {List of primary sources with key contributions}
        {Research methodology overview}

        ---
        Research conducted by AI Investigative Journalist
        New York Times Style Report
        Published: {current_date}
        Last Updated: {current_time}\
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
)

linkedin_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
    description="LinkedIn Post Generator Agent",
    tools=[DuckDuckGoTools()],
    instructions="Combine research and generate LinkedIn-ready summaries.",
    markdown=True,
    show_tool_calls=True,
)

# Agent selection function
def get_agent(agent_type: str, use_vision: bool = False):
    """
    Get the appropriate agent based on type and vision requirements.

    Args:
        agent_type (str): The type of agent requested
        use_vision (bool): Whether to use vision-enabled model for general agent

    Returns:
        Agent: The appropriate agent instance
    """
    print(f"Getting agent: type={agent_type}, use_vision={use_vision}")

    if agent_type == "general":
        agent = vision_agent if use_vision else general_agent
        print(f"Selected general agent: {agent.model.id if hasattr(agent.model, 'id') else 'unknown'}")
        return agent
    elif agent_type == "web":
        return web_agent
    elif agent_type == "youtube":
        return youtube_agent
    elif agent_type == "articles":
        return research_agent
    elif agent_type == "linkedin":
        return linkedin_agent
    elif agent_type == "finance":
        return finance_agent
    else:
        return general_agent  # Default to general agent