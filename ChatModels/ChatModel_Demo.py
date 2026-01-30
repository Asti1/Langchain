# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceHub, HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os
load_dotenv()

# model = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0, max_completion_tokens=10, openai_api_key=os.getenv("GOOGLE_API_KEY"))
# result = model.invoke("What is the capital of France?")
# print(result.content)

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="chat-completion",
)

model = ChatHuggingFace(llm=llm)
result = llm.invoke("What is the capital of France?")
print(result.content)