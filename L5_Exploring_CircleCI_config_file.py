import warnings

from utils import get_circle_api_key
from utils import get_gh_api_key
from utils import get_openai_api_key
from utils import get_repo_name
from utils import get_branch
from utils import push_files
from utils import trigger_commit_evals


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
print(course_repo)
print(course_branch)

## ------------------------------------------------------ ##
push_files(course_repo, course_branch, ["app.py", "test_assistant.py"],
           "circle_config_v1.yml")

## ------------------------------------------------------ ##
trigger_commit_evals(course_repo, course_branch, cci_api_key)

## ------------------------------------------------------ ##
push_files(course_repo, course_branch, ["app.py", "test_assistant.py"],
           "circle_config_v2.yml")

## ------------------------------------------------------ ##
trigger_commit_evals(course_repo, course_branch, cci_api_key)

## ------------------------------------------------------ ##
push_files(course_repo, course_branch,
           ["app.py", "test_assistant.py", "test_release_evals.py"],
           config = "circle_config_v3.yml")

## ------------------------------------------------------ ##
trigger_commit_evals(course_repo, course_branch, cci_api_key)

## ------------------------------------------------------ ##
push_files(course_repo, course_branch,
           ["app.py", "test_assistant.py", "test_release_evals.py"],
           config = "circle_config_v4.yml")

## ------------------------------------------------------ ##
trigger_commit_evals(course_repo, course_branch, cci_api_key)
