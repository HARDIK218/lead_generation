import datetime
import os
import requests
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()

os.environ['GROQ_API_KEY'] = "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"
os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')

# Initialize the LLM
llm = ChatGroq(temperature=0.2, model_name="llama3-70b-8192")

class SearchTools:
    @staticmethod
    def run_with_timeout(func, *args, timeout=30):
        with ThreadPoolExecutor() as executor:
            future = executor.submit(func, *args)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeoutError:
                return f"The operation timed out after {timeout} seconds."

    @tool('search facebook groups')
    def search_facebook_groups(query: str) -> str:
        """
        Use this tool to search Facebook groups. This tool returns 5 results from Facebook groups.
        """
        return SearchTools.run_with_timeout(SearchTools.search, query, limit=5)

    @tool('search linkedin groups')
    def search_linkedin_groups(query: str) -> str:
        """
        Use this tool to search LinkedIn groups. This tool returns 5 results from LinkedIn groups.
        """
        return SearchTools.run_with_timeout(SearchTools.search, query, limit=5)

    @tool('open page')
    def open_page(url: str) -> str:
        """
        Use this tool to open a webpage and get the content.
        """
        return SearchTools.run_with_timeout(lambda: WebBaseLoader(url).load())

    @staticmethod
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
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.request("POST", url, headers=headers, data=payload, timeout=25)
                response.raise_for_status()
                results = response.json().get('organic', [])

                if not results:
                    return "No results found."

                string = []
                for result in results:
                    string.append(f"{result['title']}\n{result['snippet']}\n{result['link']}\n\n")

                return f"Search results for '{query}':\n\n" + "\n".join(string)
            except requests.Timeout:
                if attempt < max_retries - 1:
                    continue  # Retry
                return "The request timed out. Moving on to the next task."
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    continue  # Retry
                return f"An error occurred: {str(e)}. Moving on to the next task."
            except Exception as e:
                return f"An unexpected error occurred: {str(e)}. Moving on to the next task."

# Define the Lead Generation Agent for Facebook
facebook_lead_generator = Agent(
    role='Facebook Lead Generator',
    goal='Identify and list potential leads from Facebook groups that are relevant to the product or business. Focus on finding groups where potential customers or business partners might be active.',
    backstory=(
        "With an understanding of social networks and group dynamics, you excel at identifying potential leads within Facebook groups. Your ability to pinpoint active and relevant groups makes you a valuable asset for generating business connections and leads."
    ),
    tools=[SearchTools.search_facebook_groups, SearchTools.open_page],
    llm=llm,
    allow_delegation=True
)

# Define the Lead Generation Agent for LinkedIn
linkedin_lead_generator = Agent(
    role='LinkedIn Lead Generator',
    goal='Identify and list potential leads from LinkedIn groups and connections that are relevant to the product or business. Focus on finding groups and connections where potential customers or business partners might be active.',
    backstory=(
        "With a strong grasp of professional networking, you excel at identifying potential leads within LinkedIn groups and connections. Your expertise in navigating LinkedIn to find relevant contacts makes you essential for generating business leads."
    ),
    tools=[SearchTools.search_linkedin_groups, SearchTools.open_page],
    llm=llm,
    allow_delegation=True
)

# Define the lead generation tasks
lead_generation_task = Task(
    description="""Identify and list potential leads from Facebook and LinkedIn groups relevant to the product or business. Focus on finding groups and connections where potential customers or business partners might be active. 

    Current date: {current_date}

    Description of the product or business for which you are doing this research: 
    <BUSINESS_DESCRIPTION>{business_description}</BUSINESS_DESCRIPTION>

    Find the most relevant groups and connections for generating business leads.
""",
    expected_output='A report with the most relevant leads that you found, including links to groups and profiles, and any other information that could be useful for the sales team.',
    tools=[],
    agent=facebook_lead_generator,
    output_file="facebook_leads.md",
)

lead_generation_task_linkedin = Task(
    description="""Identify and list potential leads from LinkedIn groups and connections relevant to the product or business. Focus on finding groups and connections where potential customers or business partners might be active. 

    Current date: {current_date}

    Description of the product or business for which you are doing this research: 
    <BUSINESS_DESCRIPTION>{business_description}</BUSINESS_DESCRIPTION>

    Find the most relevant groups and connections for generating business leads.
""",
    expected_output='A report with the most relevant leads that you found, including links to groups and profiles, and any other information that could be useful for the sales team.',
    tools=[],
    agent=linkedin_lead_generator,
    output_file="linkedin_leads.md",
)

crew = Crew(
    agents=[facebook_lead_generator, linkedin_lead_generator],
    tasks=[lead_generation_task, lead_generation_task_linkedin],
    process=Process.sequential,
)

inputs = {
    'current_date': '2024/05/22',
    'business_description': 'selling and buying e-waste for recycling',
}

result = crew.kickoff(inputs=inputs)
print(result)
