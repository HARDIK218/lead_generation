import streamlit as st
import datetime
import os
import requests
import json
from concurrent.futures import ThreadPoolExecutor
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_community.document_loaders import WebBaseLoader

# Load environment variables
load_dotenv()

# Set API keys
os.environ['GROQ_API_KEY'] = "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"
os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')

# Initialize the LLM
llm = ChatGroq(temperature=0.2, model_name="llama3-70b-8192")

# Define LeadSearchTools class for lead generation task
class LeadSearchTools:
    @tool('search_facebook_groups')
    def search_facebook_groups(query: str) -> str:
        return LeadSearchTools.search(f"site:facebook.com/groups {query}", limit=5)

    @tool('search_twitter')
    def search_twitter(query: str) -> str:
        return LeadSearchTools.search(f"site:twitter.com {query}", limit=5)

    @tool('search_news')
    def search_news(query: str) -> str:
        return LeadSearchTools.search(f"site:news.google.com {query}", limit=5)

    @tool('search_reddit')
    def search_reddit(query: str) -> str:
        return LeadSearchTools.search(f"site:reddit.com {query}", limit=5)

    @tool('open_page')
    def open_page(url: str) -> str:
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
        try:
            response = requests.request("POST", url, headers=headers, data=payload, timeout=10)
            response.raise_for_status()
            results = response.json().get('organic', [])
            if not results:
                return "No results found."

            string = []
            for result in results:
                string.append(f"{result['title']}\n{result['snippet']}\n{result['link']}\n\n")

            return f"Search results for '{query}':\n\n" + "\n".join(string)
        except requests.Timeout:
            return "The request timed out. Please try again."
        except requests.RequestException as e:
            return f"An error occurred: {str(e)}"

# Define the Lead Generation Agent
lead_generator = Agent(
    role='Lead Generator',
    goal='Identify and list potential leads from various online sources relevant to the product or business.',
    backstory=(
        "With expertise in social and professional networks, you excel at identifying potential leads within various online sources. Your ability to find relevant buyers and partners makes you invaluable for generating business leads."
    ),
    tools=[
        LeadSearchTools.search_facebook_groups,
        LeadSearchTools.search_twitter,
        LeadSearchTools.search_news,
        LeadSearchTools.search_reddit,
        LeadSearchTools.open_page
    ],
    llm=llm,
    allow_delegation=True
)

# Define the lead generation task
lead_generation_task = Task(
    description="""Identify and list potential leads from various online sources relevant to the product or business. Focus on finding groups and connections where potential customers or business partners might be active.
    Current date: {current_date}
    Description of the product or business for which you are doing this research:
    <BUSINESS_DESCRIPTION>{business_description}</BUSINESS_DESCRIPTION>
    Find the most relevant groups and connections for generating business leads.""",
    expected_output='A report with the most relevant leads that you found, including links to groups and profiles, and any other information that could be useful for the sales team.',
    tools=[],
    agent=lead_generator,
    output_file="leads_report.md",
)

# Define SearchTools class for market research task
class SearchTools:
    @tool('search internet')
    def search_internet(query: str) -> str:
        return SearchTools.search(query, limit=5)

    @tool('search instagram')
    def search_instagram(query: str) -> str:
        return SearchTools.search(f"site:instagram.com {query}", limit=5)

    @tool('open page')
    def open_page(url: str) -> str:
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

# Define Agents for market research task
market_researcher = Agent(
    role='Instagram Market Researcher',
    goal='Analyze industry trends, competitor activities, and popular hashtags on Instagram.',
    backstory=(
        "Armed with a keen eye for digital trends and a deep understanding of the Instagram landscape, you excel at uncovering actionable insights from social media data. "
        "Your analytical skills are unmatched, providing a solid foundation for strategic decisions in content creation."
    ),
    tools=[
        SearchTools.search_internet,
        SearchTools.search_instagram,
        SearchTools.open_page,
    ],
    llm=llm,
    max_iter=7,
    max_rpm=5000,
    allow_delegation=True
)

content_strategist = Agent(
    role='Instagram Content Strategist',
    goal='Develop a content calendar based on market research findings, incorporating trends, optimal posting times, and strategic content themes.',
    backstory=(
        "As a master planner with a creative spirit, you have a talent for envisioning a cohesive content strategy that resonates with audiences."
    ),
    tools=[],
    max_iter=7,
    max_rpm=5000,
    llm=llm,
    allow_delegation=True
)

visual_creator = Agent(
    role='Instagram Visual Creator',
    goal='Generate a detailed description of the images that will be used in the Instagram account during the current week tailored to the content strategy.',
    backstory=(
        "Merging creativity with technology, you use words to bring visions to life. You are great at crafting a detailed image description that can be used as a prompt for an AI-image generator."
    ),
    tools=[],
    max_iter=7,
    max_rpm=5000,
    llm=llm,
    allow_delegation=False
)

copywriter = Agent(
    role='Instagram Copywriter',
    goal='Craft engaging and relevant copy for each Instagram post, complementing the visual content and adhering to the strategic content themes.',
    backstory=(
        "With a flair for storytelling and a persuasive pen, you create narratives that captivate and engage the audience. Your words are the bridge between the brand and its followers, embodying the brand's voice in every caption and call to action."
    ),
    tools=[],
    llm=llm,
    max_iter=7,
    max_rpm=5000,
    allow_delegation=True
)

# Define tasks for market research
market_research = Task(
  description="""Investigate the latest trends, hashtags, and competitor activities on Instagram specific to the industry of this Instagram account.
    Current date: {current_date}
    Description of the instagram account for which you are doing this research:
    <INSTAGRAM_ACCOUNT_DESCRIPTION>{instagram_description}</INSTAGRAM_ACCOUNT_DESCRIPTION>
    Based on your research, determine and suggest the most relevant topics, hashtags and trends to use in the posts for next week.""",
  expected_output='A report with the most relevant information that you found, including relevant hashtags for this week\'s content, suggested focus for next week, and all other information that could be useful to the team working on content creation.',
  tools=[],
  agent=market_researcher,
)

content_strategy = Task(
  description="""Based on the market research findings, develop a detailed schedule for posting Instagram posts over the next three days.
    The schedule should cover:
    - Themes and post ideas for each day
    - Detailed description of content (including visuals and text)
    - Best hashtags to use for each post
    - Suggested posting times""",
  expected_output='A detailed schedule formatted as markdown, covering the next three days. Each day should include the post theme, content, hashtags, and suggested posting time.',
  tools=[],
  agent=content_strategist,
  dependencies=[market_research],
  output_file="content_schedule.md"
)

visual_creation = Task(
  description="""Generate a detailed description of the images that will be used in the Instagram account during the current week tailored to the content strategy.
    Current date: {current_date}
    Description of the instagram account for which you are doing this research:
    <INSTAGRAM_ACCOUNT_DESCRIPTION>{instagram_description}</INSTAGRAM_ACCOUNT_DESCRIPTION>
    Based on the content strategy provided in the schedule, create detailed descriptions for each image required.""",
  expected_output='A list of detailed descriptions for each image to be used for the week.',
  tools=[],
  agent=visual_creator,
  dependencies=[content_strategy]
)

copywriting = Task(
  description="""Craft engaging and relevant copy for each Instagram post, complementing the visual content and adhering to the strategic content themes.
    Current date: {current_date}
    Description of the instagram account for which you are doing this research:
    <INSTAGRAM_ACCOUNT_DESCRIPTION>{instagram_description}</INSTAGRAM_ACCOUNT_DESCRIPTION>
    Based on the content strategy and visual descriptions, create a copy for each post. The copy should include any relevant hashtags and call-to-actions.""",
  expected_output='A list of captions and copy for each post, ready to be used.',
  tools=[],
  agent=copywriter,
  dependencies=[visual_creation],
  output_file="captions.md"
)

# Function to execute lead generation task
def execute_lead_generation_task():
    with ThreadPoolExecutor() as executor:
        future = executor.submit(lead_generation_task.execute)
        return future.result()

# Function to execute market research task
def execute_market_research_task():
    with ThreadPoolExecutor() as executor:
        future = executor.submit(copywriting.execute)
        return future.result()

# Streamlit UI
st.title('Lead Generation and Market Research Task Executor')

task_choice = st.selectbox('Choose a task to execute:', ['Lead Generation', 'Market Research'])

if task_choice == 'Lead Generation':
    st.header('Lead Generation Task')
    business_description = st.text_input('Enter a description of the product or business:')
    if st.button('Execute Lead Generation Task'):
        if business_description:
            lead_generation_task.variables['current_date'] = datetime.datetime.now().strftime("%Y-%m-%d")
            lead_generation_task.variables['business_description'] = business_description
            result = execute_lead_generation_task()
            st.markdown(result)
        else:
            st.warning('Please enter a business description.')
else:
    st.header('Market Research Task')
    instagram_description = st.text_input('Enter a description of the Instagram account:')
    if st.button('Execute Market Research Task'):
        if instagram_description:
            copywriting.variables['current_date'] = datetime.datetime.now().strftime("%Y-%m-%d")
            copywriting.variables['instagram_description'] = instagram_description
            result = execute_market_research_task()
            st.markdown(result)
        else:
            st.warning('Please enter an Instagram account description.')
