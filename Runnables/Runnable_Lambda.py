from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnableLambda, RunnablePassThrough

prompt1 = PromptTemplate(
    template= "Write a joke about {topic}",
    input_variables = ['topic']
)

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

joke_gen = RunnableSequence(prompt1, model, parser)
def word_count(text):
    return len(text.split())

parallel_chain = RunnableParallel({
    'joke': RunnablePassThrough(),
    'word_count': RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_gen, parallel_chain)
print(final_chain.invoke({'topic': 'AI'}))