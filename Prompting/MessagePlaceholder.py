from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import json
from langchain_core.messages import HumanMessage, AIMessage
chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

with open("chat_history.json") as f:
    data = json.load(f)
chat_history = [
    HumanMessage(content=m["content"]) if m["role"]=="human"
    else AIMessage(content=m["content"])
    for m in data
]

prompt = chat_template.invoke({'chat_history':chat_history, 'query':'Where is my refund'})
print(prompt)