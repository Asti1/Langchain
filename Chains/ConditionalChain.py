from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
load_dotenv()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description= "Give the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object = Feedback)
prompt1 = PromptTemplate(
    template = 'Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}',
    input_variables = ['feedback'],
    partial_variables = {'format_instruction': parser2.get_format_instructions()}
)
prompt2 = PromptTemplate(
    template = 'Write an appropriate response to this positive feedback \n {feedback}',
    input_variables = ['feedback']
)
prompt3 = PromptTemplate(
    template = 'Write an appropriate response to this negative feedback \n {feedback}',
    input_variables = ['feedback']
)
llm1 = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    task='text-generation'
)
llm2 = HuggingFaceEndpoint(
    repo_id = "TheBloke/Llama-2-Pro-Mistral-7B-GGUF",
    task='text-generation'
)
model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)
parser = StrOutputParser()

classifier_chain = prompt1 | model1 | parser2
branch_chain = RunnableBranch(
    (lambda x:x.sentiment == "positive", prompt2 | model1 | parser),
    (lambda x:x.sentiment== "negative", prompt3 | model1 | parser),
    RunnableLambda(lambda x: "Invalid sentiment")
)

chain = classifier_chain | branch_chain
result = chain.invoke({'feedback': "The product quality is really bad"})
print(result)