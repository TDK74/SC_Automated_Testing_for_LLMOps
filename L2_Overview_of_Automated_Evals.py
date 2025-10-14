import os
import warnings

from app import assistant_chain
from app import system_message

from utils import get_circle_api_key
from utils import get_gh_api_key
from utils import get_openai_api_key
from utils import get_repo_name
from utils import get_branch
from utils import push_files
from utils import trigger_commit_evals

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv())
warnings.filterwarnings('ingnore')

## ------------------------------------------------------ ##
cci_api_key = get_circle_api_key()

## ------------------------------------------------------ ##
gh_api_key = get_gh_api_key()

## ------------------------------------------------------ ##
openai_api_key = get_openai_api_key()

## ------------------------------------------------------ ##
course_repo = get_repo_name()
course_repo   # print()

## ------------------------------------------------------ ##
course_branch = get_branch()
course_branch   # print()
## ------------------------------------------------------ ##
human_template = "{question}"

## ------------------------------------------------------ ##
quiz_bank = """1. Subject: Leonardo DaVinci
                   Categories: Art, Science
                   Facts:
                    - Painted the Mona Lisa
                    - Studied zoology, anatomy, geology, optics
                    - Designed a flying machine

                2. Subject: Paris
                   Categories: Art, Geography
                   Facts:
                    - Location of the Louvre, the museum where the Mona Lisa is displayed
                    - Capital of France
                    - Most populous city in France
                    - Where Radium and Polonium were discovered by scientists Marie and \
                        Pierre Curie

                3. Subject: Telescopes
                   Category: Science
                   Facts:
                    - Device to observe different objects
                    - The first refracting telescopes were invented in the Netherlands in \
                        the 17th Century
                    - The James Webb space telescope is the largest telescope in space. \
                        It uses a gold-berillyum mirror

                4. Subject: Starry Night
                   Category: Art
                   Facts:
                    - Painted by Vincent van Gogh in 1889
                    - Captures the east-facing view of van Gogh's room in Saint-Rémy-de-Provence

                5. Subject: Physics
                   Category: Science
                   Facts:
                    - The sun doesn't change color during sunset.
                    - Water slows the speed of light
                    - The Eiffel Tower in Paris is taller in the summer than the winter due \
                        to expansion of the metal."""

## ------------------------------------------------------ ##
delimiter = "####"

prompt_template = f"""
                Follow these steps to generate a customized quiz for the user.
                The question will be delimited with four hashtags i.e {delimiter}

                The user will provide a category that they want to create a quiz for. Any \
                questions included in the quiz should only refer to the category.

                Step 1:{delimiter} First identify the category user is asking about from \
                the following list:
                * Geography
                * Science
                * Art

                Step 2:{delimiter} Determine the subjects to generate questions about. \
                The list of topics are below:

                {quiz_bank}

                Pick up to two subjects that fit the user's category.

                Step 3:{delimiter} Generate a quiz for the user. Based on the selected \
                subjects generate 3 questions for the user using the facts about the subject.

                Use the following format for the quiz:
                Question 1:{delimiter} <question 1>

                Question 2:{delimiter} <question 2>

                Question 3:{delimiter} <question 3>

                """

## ------------------------------------------------------ ##
chat_prompt = ChatPromptTemplate.from_messages([("human", prompt_template)])
chat_prompt   # print()

## ------------------------------------------------------ ##
llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0)
llm   # print()

## ------------------------------------------------------ ##
output_parser = StrOutputParser()
output_parser   # print()

## ------------------------------------------------------ ##
chain = chat_prompt | llm | output_parser
chain   # print()

## ------------------------------------------------------ ##
def assistant_chain(system_message, human_template = "{question}",
                    llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0),
                    output_parser = StrOutputParser()):
    chat_prompt = ChatPromptTemplate.from_messages([("system", system_message),
                                                    ("human", human_template),
                                                    ])

    return chat_prompt | llm | output_parser

## ------------------------------------------------------ ##
def eval_expected_words(system_message, question, expected_words, human_template = "{question}",
                        llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0),
                        output_parser = StrOutputParser()):
    assistant = assistant_chain(system_message, human_template, llm, output_parser)

    answer = assistant.invoke({"question" : question})
    print(answer)

    assert any(word in answer.lower() for word in expected_words), \
               f"Expected the assistant questions to include '{expected_words}', but it did not"

## ------------------------------------------------------ ##
question = "Generate a quiz about science"
expected_words = ["davinci", "telescope", "physics", "curie"]

## ------------------------------------------------------ ##
eval_expected_words(prompt_template, question, expected_words)

## ------------------------------------------------------ ##
def evaluate_refusal(system_message, question, decline_response, human_template = "{question}",
                    llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0),
                    output_parser = StrOutputParser()):
    assistant = assistant_chain(human_template, system_message, llm, output_parser)

    answer = assistant.invoke({"question" : question})
    print(answer)

    assert decline_response.lower() in answer.lower(), \
        f"Expected the bot to decline with '{decline_response}' got {answer}"

## ------------------------------------------------------ ##
question = "Generate a quiz about Rome."
decline_response = "I am sorry."

## ------------------------------------------------------ ##
evaluate_refusal(prompt_template, question, decline_response)

## ------------------------------------------------------ ##
delimiter = "####"

quiz_bank = """1. Subject: Leonardo DaVinci
                   Categories: Art, Science
                   Facts:
                    - Painted the Mona Lisa
                    - Studied zoology, anatomy, geology, optics
                    - Designed a flying machine

                2. Subject: Paris
                   Categories: Art, Geography
                   Facts:
                    - Location of the Louvre, the museum where the Mona Lisa is displayed
                    - Capital of France
                    - Most populous city in France
                    - Where Radium and Polonium were discovered by scientists Marie \
                        and Pierre Curie

                3. Subject: Telescopes
                   Category: Science
                   Facts:
                    - Device to observe different objects
                    - The first refracting telescopes were invented in the Netherlands in \
                        the 17th Century
                    - The James Webb space telescope is the largest telescope in space. \
                        It uses a gold-berillyum mirror

                4. Subject: Starry Night
                   Category: Art
                   Facts:
                    - Painted by Vincent van Gogh in 1889
                    - Captures the east-facing view of van Gogh's room in Saint-Rémy-de-Provence

                5. Subject: Physics
                   Category: Science
                   Facts:
                    - The sun doesn't change color during sunset.
                    - Water slows the speed of light
                    - The Eiffel Tower in Paris is taller in the summer than the winter due \
                        to expansion of the metal.
                """

system_message = f"""
                Follow these steps to generate a customized quiz for the user.
                The question will be delimited with four hashtags i.e {delimiter}

                The user will provide a category that they want to create a quiz for. Any \
                questions included in the quiz should only refer to the category.

                Step 1:{delimiter} First identify the category user is asking about from \
                the following list:
                * Geography
                * Science
                * Art

                Step 2:{delimiter} Determine the subjects to generate questions about. \
                The list of topics are below:

                {quiz_bank}

                Pick up to two subjects that fit the user's category.

                Step 3:{delimiter} Generate a quiz for the user. Based on the selected \
                subjects generate 3 questions for the user using the facts about the subject.

                Use the following format for the quiz:
                Question 1:{delimiter} <question 1>

                Question 2:{delimiter} <question 2>

                Question 3:{delimiter} <question 3>

                Additional rules:

                - Only use explicit matches for the category, if the category is not an exact \
                    match to categories in the quiz bank, answer that you do not have information.
                - If the user asks a question about a subject you do not have information about \
                    in the quiz bank, answer "I'm sorry I do not have information about that".
                """

"""
Helper functions for writing the test cases
"""


def assistant_chain(system_message = system_message, human_template = "{question}",
                    llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0),
                    output_parser = StrOutputParser()):
        chat_prompt = ChatPromptTemplate.from_messages([("system", system_message),
                                                       ("human", human_template),
                                                       ])

        return chat_prompt | llm | output_parser

## ------------------------------------------------------ ##
def eval_expected_words(system_message, question, expected_words, human_template = "{question}",
                        llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0),
                        output_parser = StrOutputParser()):
    assistant = assistant_chain(system_message)

    answer = assistant.invoke({"question" : question})
    print(answer)

    assert any(word in answer.lower() for word in expected_words), \
               f"Expected the assistant questions to include '{expected_words}', but it did not"


def evaluate_refusal(system_message, question, decline_response, human_template = "{question}",
                    llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0),
                    output_parser = StrOutputParser()):
    assistant = assistant_chain(human_template, system_message, llm, output_parser)

    answer = assistant.invoke({"question" : question})
    print(answer)

    assert decline_response.lower() in answer.lower(), \
        f"Expected the bot to decline with '{decline_response}' got {answer}"


"""
Test cases
"""


def test_science_quiz():
    question = "Generate a quiz about science."
    expected_subjects = ["davinci", "telescope", "physics", "curie"]

    eval_expected_words(system_message, question, expected_subjects)


def test_geography_quiz():
    question = "Generate a quiz about geography."
    expected_subjects = ["paris", "france", "louvre"]

    eval_expected_words(system_message, question, expected_subjects)


def test_refusal_rome():
     question = "Help me create a quiz about Rome."
     decline_response = "I am sorry."

     evaluate_refusal(system_message, question, decline_response)

## ------------------------------------------------------ ##
push_files(course_repo, course_branch, ["app.py", "test_assistant.py"])

## ------------------------------------------------------ ##
trigger_commit_evals(course_repo, course_branch, cci_api_key)
