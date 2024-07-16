import datetime
import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import requests
import json
from langchain.tools import tool
from langchain_community.document_loaders import WebBaseLoader

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = "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"
os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')

# Initialize the LLM
llm = ChatGroq(temperature=0.2, model_name="llama3-70b-8192")


# --------- Tools ---------#
class SearchTools:

    @tool('search internet')
    def search_internet(query: str) -> str:
        """
        Use this tool to search the internet for information. This tools returns 5 results from Google search engine.
        """
        return SearchTools.search(query, limit=5)

    @tool('search instagram')
    def search_instagram(query: str) -> str:
        """
        Use this tool to search Instagram. This tools returns 5 results from Instagram pages.
        """
        return SearchTools.search(f"site:instagram.com {query}", limit=5)

    @tool('open page')
    def open_page(url: str) -> str:
        """
        Use this tool to open a webpage and get the content.
        """
        loader = WebBaseLoader(url)
        return loader.load()

    def search(query, limit=5):
        url = "https://google.serper.dev/search"
        payload = json.dumps({
            "q": query,
            "num": limit,
        })
        headers = {
            'X-API-KEY': os.getenv("SERPER_API_KEY"),
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        results = response.json()['organic']

        string = []
        for result in results:
            string.append(f"{result['title']}\n{result['snippet']}\n{result['link']}\n\n")

        return f"Search results for '{query}':\n\n" + "\n".join(string)


# --------- Defining Agents ---------#

# Define the Instagram Market Researcher agent
market_researcher = Agent(
    role='Instagram Market Researcher',
    goal='Analyze industry trends, competitor activities, and popular hashtags on Instagram. Perform research on the latest trends, hashtags, and competitor activities on Instagram using your Search tools.',
    backstory=(
        "Armed with a keen eye for digital trends and a deep understanding of the Instagram landscape, you excel at uncovering actionable insights from social media data. "
        "Your analytical skills are unmatched, providing a solid foundation for strategic decisions in content creation. You are great at identifying the latest trends and the best hashtags for a given campaign."
    ),
    tools=[
        SearchTools.search_internet,
        SearchTools.search_instagram,
        SearchTools.open_page,
    ],
    llm=llm,
    max_iter=4,
    max_rpm=4000,
    allow_delegation=True
)

# Define the Instagram Content Strategist agent
content_strategist = Agent(
    role='Instagram Content Strategist',
    goal='Develop a content calendar based on market research findings, incorporating trends, optimal posting times, and strategic content themes for next three days.',
    backstory=(
        "As a master planner with a creative spirit, you have a talent for envisioning a cohesive content strategy that resonates with audiences. "
        "Your expertise in aligning content with brand voice and audience interests has consistently driven engagement and growth."
    ),
    tools=[],
    max_iter=4,
    max_rpm=4000,
    llm=llm,
    allow_delegation=True
)

# Define the Instagram Copywriter agent
writer = Agent(
    role='Instagram Copywriter',
    goal='Craft engaging and relevant copy for each Instagram post, complementing the visual content and adhering to the strategic content themes.',
    backstory=(
        "With a flair for storytelling and a persuasive pen, you create narratives that captivate and engage the audience. Your words are the bridge between the brand and its followers, embodying the brand's voice in every caption and call to action. "
        "Your writing is not only engaging but also incorporates all the SEO techniques, such as seamlessly using top keywords given to you and adding the best hashtags that are trending at the moment."
    ),
    tools=[],
    llm=llm,
    max_iter=4,
    max_rpm=4000,
    allow_delegation=True
)

# ---------- Defining task ----------#

market_research = Task(
    description="""Investigate the latest trends, hashtags, and competitor activities on Instagram specific to the industry of this Instagram account. Focus on gathering data that reveals what content performs well in the current year, identifying patterns, preferences, and emerging trends. 

    Current date: {current_date}

    Description of the instagram account for which you are doing this research: 
    <INSTAGRAM_ACCOUNT_DESCRIPTION>{instagram_description}</INSTAGRAM_ACCOUNT_DESCRIPTION>

    Based on your research, determine and suggest the most relevant topics, hashtags and trends to use in the posts for 3 DAYS.
    """,
    expected_output='A report with the most relevant information that you found, including relevant hashtags for this week\'s content, suggested focus for next week, and all other information that could be useful to the team working on content creation.',
    tools=[],
    agent=market_researcher,
    output_file="market_research.md",
)

content_strategy = Task(
    description="""Based on the market research findings, develop a detailed schedule for posting Instagram posts over the next three days. The schedule should include the theme for each day, DETAILED CONTENT IDEA, and the most relevant hashtags to use for each post. Focus on what will improve customer engagement, including optimal posting times and strategies to increase interaction.

    The schedule should cover:
    - Themes and post ideas for each day
    - Detailed description of content
    - Best hashtags to use for each post
    - Suggested posting times
    """,
    expected_output='A detailed schedule formatted as markdown, covering the next three days. Each day should include the theme, content ideas, hashtags, and suggested posting times to improve engagement.',
    tools=[],
    agent=content_strategist,
)

writing = Task(
    description="""Write captivating and relevant copy for each Instagram post of the week, aligning to the strategic themes of the content calendar. The copy should engage the audience, embody the brand's voice, and encourage interaction. The copy should also be SEO-friendly and incorporate the relevant keywords and hashtags contained in the content schedule. 

    Consider the following guidelines when writing the copy:
    - Keep the copy concise and engaging.
    - Include a call to action where appropriate.
    - Use relevant keywords and hashtags.
    - Ensure the copy aligns with the brand's voice and tone.
    """,
    expected_output='A document formatted as markdown, with several sections. Each section should contain the copy for a single Instagram post, along with the relevant hashtags and calls to action. The copy should be engaging, on-brand, and aligned with the content calendar.',
    tools=[],
    agent=writer,
)

crew = Crew(
    agents=[market_researcher, content_strategist,writer],
    tasks=[market_research, content_strategy, writing],
    process=Process.sequential,
)

# Streamlit UI
st.title('Instagram Content Strategy Generator')

instagram_description = st.text_input('Instagram Account Description', 'Electronic waste')
current_date = st.date_input('Current Date', datetime.date.today())

if st.button('Generate Strategy'):
    inputs = {
        'current_date': current_date.strftime('%Y/%m/%d'),
        'instagram_description': instagram_description
    }

    with st.spinner('Generating strategy...'):
        crew.kickoff(inputs=inputs)

    # Read and display the content of market_research.md
    with open('market_research.md', 'r') as file:
        market_research_content = file.read()

    st.markdown("### Market Research Report")
    st.markdown(market_research_content)

    st.markdown("### Final Result")
    result = crew.kickoff(inputs=inputs)
    st.markdown(result)
