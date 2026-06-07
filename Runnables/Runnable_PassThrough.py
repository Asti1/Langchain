from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassThrough

prompt1 = PromptTemplate(
    template= "Write a joke about {topic}",
    input_variables = ['topic']
)
prompt2 = PromptTemplate(
    template= "Explain the joke: {text}",
    input_variables = ['text']
)
llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    task='text-generation'
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()
jokegen = RunnableSequence(prompt1, model, parser)
parallel_chain = RunnableParallel({
    'joke': RunnablePassThrough(),
    'explaination': RunnableSequence(prompt2, model, parser)
})
final_chain = RunnableSequence(jokegen, parallel_chain)

print(final_chain.invoke({'topic': 'AI'}))
