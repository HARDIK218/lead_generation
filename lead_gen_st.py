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

class LeadSearchTools:

    @tool('search_facebook_groups')
    def search_facebook_groups(query: str) -> str:
        """Search Facebook groups. This tool returns 5 results from Facebook groups."""
        return LeadSearchTools.search(f"site:facebook.com/groups {query}", limit=5)

    @tool('search_twitter')
    def search_twitter(query: str) -> str:
        """Search Twitter. This tool returns 5 results from Twitter."""
        return LeadSearchTools.search(f"site:twitter.com {query}", limit=5)

    @tool('search_news')
    def search_news(query: str) -> str:
        """Search news articles. This tool returns 5 results from news sources."""
        return LeadSearchTools.search(f"site:news.google.com {query}", limit=5)

    @tool('search_reddit')
    def search_reddit(query: str) -> str:
        """Search Reddit. This tool returns 5 results from Reddit."""
        return LeadSearchTools.search(f"site:reddit.com {query}", limit=5)

    @tool('open_page')
    def open_page(url: str) -> str:
        """Open a webpage and get the content."""
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
    goal='Identify and list potential leads from various online sources relevant to the product or business. Focus on finding groups and connections where potential customers or business partners might be active.',
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

    Find the most relevant groups and connections for generating business leads.
""",
    expected_output='A report with the most relevant leads that you found, including links to groups and profiles, and any other information that could be useful for the sales team.',
    tools=[],
    agent=lead_generator,
)

def execute_lead_generation_task(inputs):
    # Initialize and run the crew with the lead generation task
    crew = Crew(
        agents=[lead_generator],
        tasks=[lead_generation_task],
        process=Process.sequential,
    )
    result = crew.kickoff(inputs=inputs)
    return result

# Streamlit UI
st.title('Lead Generation Task Executor')

business_description = st.text_input('Enter a description of the product or business:')
current_date = st.date_input('Enter the current date:', datetime.date.today())

if st.button('Execute Lead Generation Task'):
    if business_description:
        inputs = {
            'current_date': current_date.strftime('%Y/%m/%d'),
            'business_description': business_description,
        }
        with st.spinner('Executing task...'):
            result = execute_lead_generation_task(inputs)
        st.markdown(result)
    else:
        st.warning('Please enter a business description.')
