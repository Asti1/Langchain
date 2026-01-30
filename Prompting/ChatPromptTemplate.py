from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import SystemMessage, HumanMessage,AIMessage
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

chat_template = ChatPromptTemplate([
    ('system', 'You are a helful {domain} expert'),
    ('human', 'Explain in simple terms what is {topic}')

])

prompt = chat_template.invoke({'domain': 'football', 'topic':'red card'})

print(prompt)