import warnings

from utils import get_circle_api_key
from utils import get_gh_api_key
from utils import get_openai_api_key
from utils import get_repo_name
from utils import get_branch
from utils import push_files
from utils import trigger_release_evals

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser


warnings.filterwarnings('ignore')

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
delimiter = "####"

## ------------------------------------------------------ ##
eval_system_prompt = f"""You are an assistant that evaluates \
                      whether or not an assistant is producing valid quizzes.
                      The assistant should be producing output in the \
                      format of Question N:{delimiter} <question N>?"""

## ------------------------------------------------------ ##
llm_response = """
                Question 1:#### What is the largest telescope in space called and \
                what material is its mirror made of?

                Question 2:#### True or False: Water slows down the speed of light.

                Question 3:#### What did Marie and Pierre Curie discover in Paris?
                """

## ------------------------------------------------------ ##
eval_user_message = f"""You are evaluating a generated quiz based on the context \
                    that the assistant uses to create the quiz.
                    Here is the data:
                        [BEGIN DATA]
                        ************
                        [Response]: {llm_response}
                        ************
                        [END DATA]

                    Read the response carefully and determine if it looks like \
                    a quiz or test. Do not evaluate if the information is correct
                    only evaluate if the data is in the expected format.

                    Output Y if the response is a quiz, \
                    output N if the response does not look like a quiz.
                    """

## ------------------------------------------------------ ##
eval_prompt = ChatPromptTemplate.from_messages([("system", eval_system_prompt),
                                                ("human", eval_user_message), ])

## ------------------------------------------------------ ##
llm = ChatOpenAI(model = "gpt-3.5-turbo",
                 temperature = 0)

## ------------------------------------------------------ ##
output_parser = StrOutputParser()

## ------------------------------------------------------ ##
eval_chain = eval_prompt | llm | output_parser

## ------------------------------------------------------ ##
eval_chain.invoke({})

## ------------------------------------------------------ ##
def create_eval_chain(agent_response,
                    llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0),
                    output_parser = StrOutputParser()):
  delimiter = "####"

  eval_system_prompt = f"""You are an assistant that evaluates whether or not an assistant \
                    is producing valid quizzes.
                    The assistant should be producing output in the format of Question \
                    N:{delimiter} <question N>?"""

  eval_user_message = f"""You are evaluating a generated quiz based on the context that the \
                    assistant uses to create the quiz.
                    Here is the data:
                        [BEGIN DATA]
                        ************
                        [Response]: {agent_response}
                        ************
                        [END DATA]

                    Read the response carefully and determine if it looks like a quiz or test. \
                    Do not evaluate if the information is correct only evaluate if the data is \
                    in the expected format.

                    Output Y if the response is a quiz, output N if the response does not look \
                    like a quiz.
                    """

  eval_prompt = ChatPromptTemplate.from_messages([
                                                ("system", eval_system_prompt),
                                                ("human", eval_user_message),
                                                ])

  return eval_prompt | llm | output_parser

## ------------------------------------------------------ ##
known_bad_result = "There are lots of interesting facts. \
                    Tell me more about what you'd like to know"

## ------------------------------------------------------ ##
bad_eval_chain = create_eval_chain(known_bad_result)

## ------------------------------------------------------ ##
bad_eval_chain.invoke({})

## ------------------------------------------------------ ##
push_files(course_repo, course_branch,
           ["app.py", "test_release_evals.py", "test_assistant.py"],
           config = "circle_config.yml")

## ------------------------------------------------------ ##
trigger_release_evals(course_repo, course_branch,
                    ["app.py", "test_assistant.py", "test_release_evals.py"],
                    cci_api_key)
