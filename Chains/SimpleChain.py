from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

prompt1 = PromptTemplate(
    template = 'Generate 5 interesting facts about {topic}',
    input_variables = ['topic']
)

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    task='text-generation'
)
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()
chain = prompt | model | parser
# result = chain.invoke({'topic': 'Langchain'})
# print(result)
chain.get_graph().print_ascii()