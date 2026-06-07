from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnableBranch, RunnablePassThrough

prompt1 = PromptTemplate(
    template= "Write a detailed report about {topic}",
    input_variables= ['topic']
)
prompt2 = PromptTemplate(
    template= "Summarize the following text \n {text}",
    input_variables = ['text']
)

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    task='text-generation'
)
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

report_gen = RunnableSequence(prompt1, model, parser)
branch_chain = RunnableBranch(
    (lambda x: len(x.split())>500, RunnableSequence(prompt2, model, parser)),
    RunnablePassThrough()
)

final_chain = RunnableSequence(report_gen, branch_chain)
print(final_chain.invoke({'topic': 'AI'}))
