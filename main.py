import datetime
import json
import random
from bs4 import BeautifulSoup
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.utilities import DuckDuckGoSearchAPIWrapper
import requests

# config

RESULTS_PER_QUESTION_SEARCH = 3

RESEARCH_QUESTIONS_COUNT = 3

search_api = DuckDuckGoSearchAPIWrapper()

model = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=1)

output_parser = StrOutputParser()


def search_internet_for_websites(
    question: str, results_count: int = RESULTS_PER_QUESTION_SEARCH
) -> str:
    results = search_api.results(question, results_count)
    print("\n\nSearch results \n", json.dumps(results))
    return [result["link"] for result in results]


def scrape_website(url: str) -> str:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    contents = (
        soup.get_text()
        .replace("\n", " ")
        .replace("  ", " ")
        .replace("  ", " ")
        .replace("  ", " ")
    )
    print(f"\n\nScraped content from {url}\n\n", contents)
    return contents


summarize_content_prompt = ChatPromptTemplate.from_template(
    '{context}\n Using the above text, summarize it based on the following task or query: "{question}".\n If the '
    f"query cannot be answered using the text, YOU MUST summarize the text in short.\n Include all factual "
    f"information such as numbers, stats, quotes, etc if available. "
    "DO NOT ANSWER IN MARKDOWN SYNTAX. \n"
)

generate_questions_prompt = ChatPromptTemplate.from_template(
    'You are a RESTApi that creates {question_count} google search queries to search online that form an objective opinion from the following: "{question}"'
    f'You always respond with an RFC 8259 complaint Json list of strings in the following format: ["query 1", "query 2", "query 3"].'
    "DO NOT ANSWER IN MARKDOWN SYNTAX. and DO NOT DEVIATE FROM THE OUTPUT FORMAT \n"
)

refinement_prompt = ChatPromptTemplate.from_template(
    'Information: """{summary}"""\n\n'
    f"Using the above information, answer the following"
    ' query or task: "{question}" in a detailed report --'
    " The report should focus on the answer to the query, should be well structured, informative,"
    f" in depth and comprehensive, with facts and numbers if available and a minimum of 10000 words.\n"
    "You should strive to write the report as long as you can using all relevant and necessary information provided.\n"
    "You must write the report with markdown syntax.\n "
    f"Use an unbiased and journalistic tone. \n"
    f"The title must be SEO friendly. \n"
    "You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.\n"
    f"You MUST write all used source urls at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each.\n"
    f"You MUST write the report in the apa format.\n "
    f"You MUST Cite search results using inline notations. Only cite the most \
            relevant results that answer the query accurately. Place these citations at the end \
            of the sentence or paragraph that reference them.\n"
    f"Please do your best, this is very important to my career.\n "
    f"Answer in the language of the question. "
)


question_to_research = "What are the 10 best A.I pdf reader apps and services on the market. What features do they have, how are they priced and what are their marketting strategies?"

do_research_chain = (
    # We start by generating research questions
    generate_questions_prompt
    | model
    | output_parser  # => '["question 1", "question 2", "question 3", ...]'
    # | (lambda input_data: print(input_data))
    | json.loads
    | (lambda questions: [{"question": question} for question in questions])
    # We then search the internet for websites that answer the questions
    | (
        RunnablePassthrough.assign(
            urls=lambda input_dict: search_internet_for_websites(input_dict["question"])
        )
        | (
            lambda input_dict: [
                {"question": input_dict["question"], "url": url}
                for url in input_dict["urls"]
            ]
        )
    ).map()  # => '[[{"question": "question 1", "url": "url 1"}, ...], ...]'
    | (
        lambda outerArray: [
            question_dict for innerArray in outerArray for question_dict in innerArray
        ]
    )  # => '[{"question": "question 1", "url": "url 1"}, {"question": "question 1", "url": "url 2"}, ...]'

    # We then scrape the websites and summarize them
    | (
        RunnablePassthrough.assign(
            summary=(
                RunnablePassthrough.assign(
                    context=lambda prompt_variables: scrape_website(
                        prompt_variables["url"]
                    )
                )  # => '[{"question": "question 1", "url": "url 1", "context": "context 1"}, ...]'
                | summarize_content_prompt
                | model
                | output_parser
            )
        ).map()  # => '[{"question": "question 1", "url": "url 1", "summary": "summary 1"}, ...]'
        | (
            lambda input_dict: [
                f"Research question: {data['question']} \n Researched Website: {data['url']} \n Research Results: {data['summary']}\n\n"
                for data in input_dict
            ]
        )
        | (lambda data: {"summary": "".join(data), "question": question_to_research})
        # We then refine the summaries into a full fledged report
        | refinement_prompt
        | model
        | output_parser
    )
)

output = do_research_chain.invoke(
    {
        "question": question_to_research,
        "question_count": RESEARCH_QUESTIONS_COUNT,
    }
)

with open(f"{question_to_research.replace(' ', '_')[:50]}.md", "w") as f:
    print(output)
    f.write(output)
