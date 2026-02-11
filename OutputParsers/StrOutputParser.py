from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template= "Write a detailed report on {topic}",
    input_variables = ['topic']
)
template2 = PromptTemplate(
    template= "Write a 5 line summary on {text}",
    input_variables = ['text']
)

# prompt1 = template1.invoke({'topic':"Langchain"})
# result1 = model.invoke(prompt1)
# prompt2 = template2.invoke({'text':result1.content})
# result2 = model.invoke(prompt2)
# print(result2.content)

# Using stroutputparser
parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({'topic': 'Langchain'})
print(result)