import os
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model = os.environ["MODEL"],
    base_url=os.environ["MINIMAX_BASE_URL"],
    temperature=0,
)
