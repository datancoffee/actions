from core.inference.huggingface import InferWithHuggingface
from core.readers.github import ReadGithubIssue
from core.inference.llama3instruct import InferWithLlama3Instruct
from core.transforms.github import GithubIssueToChat
from core.writers.dlt import WriteFile
import os

# Get the Huggingface token
HF_TOKEN = os.getenv("HF_TOKEN")



# Create a Github reader action that will read from Github and store in memory (readitems attributes of the class)
# This reader will read the main issue and all comments for a specific issue number
read_issue = ReadGithubIssue()
read_issue(repo_owner="dlt-hub", repo="dlt", issue_number=933)
issues= read_issue.readitems['issues']
comments = read_issue.readitems['comments']

# Start building inputs into an LLM. Issues and comments get converted into a chat-like thread
# Each message in the chat are tagged with one of 3 roles: system, assistant, user
# The first message is by 'system' and instructs our LLM how to behave
messages = []
messages.append({"role": "system", "content": "You are a coding assistant that answers user questions posted to GitHub!"})

# Now convert all comments from Github that we read earlier
convert_gh_issue = GithubIssueToChat()
chat = convert_gh_issue(issues=issues,comments = comments)
messages.extend(chat)

# Pass this chat message list to the Llama3 model and get a response
infer_with_llama3 = InferWithLlama3Instruct(HF_TOKEN, device="mps")
response = infer_with_llama3(messages)

# Check that our response is safe to use at work.
# Generate an NSFW score using a model on Huggingface
filter_nsfw = InferWithHuggingface(
    task="text-classification", model="michellejieli/NSFW_text_classifier", device="mps")
nsfw_score = filter_nsfw(response)[0]

# Start preparing outputs.
response_message = {"role": "assistant", "content":response}
response_message = {**response_message, **nsfw_score}
messages.append(response_message)

# Write the last response and the full chat to local files
write_last_response = WriteFile(
    "github_bot", bucket_url="./out/gh_bot_last_response")
write_last_response([response_message],loader_file_format="jsonl")

write_full_chat = WriteFile(
    "github_bot", bucket_url="./out/gh_bot_full_chat")
write_full_chat(messages,loader_file_format="jsonl")


