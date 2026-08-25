import os
from langchain_openai import ChatOpenAI

# Local OpenAI-compatible servers (e.g. MLX, llama.cpp, vLLM) often ignore
# the API key, so users frequently leave OPENAI_API_KEY unset. An empty
# string trips langchain-openai's auth resolver into an async-credentials
# lookup path, which breaks synchronous invoke(). Provide a harmless
# placeholder whenever the key is missing or empty.

llm = ChatOpenAI(
    model=os.environ["MODEL"],
    base_url=os.environ["BASE_URL"],
    temperature=0,
)
