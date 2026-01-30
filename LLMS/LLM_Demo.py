from langchain_openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

llm = OpenAI(model='gpt-3.5-turbo-instruct', openai_api_key=os.getenv("OPEN_AI_API"))
result = llm.invoke("What is the capital of France?")

print(result)