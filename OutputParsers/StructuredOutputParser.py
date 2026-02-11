from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='fact1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact3', description='Fact 3 about the topic'),
]
parser = StructuredOutputParser.from_response_schemas(schema)
template1 = PromptTemplate(
    template= "Give 3 facts about the {topic} \n {format_instruction}",
    input_variables = ['topic'],
    partial_variables = {
        'format_instruction': parser.get_format_instructions()
    }
)
# template2 = PromptTemplate(
#     template= "Write a 5 line summary on {text}",
#     input_variables = ['text']
# )

# prompt1 = template1.invoke({'topic':"Langchain"})
# result1 = model.invoke(prompt1)
# prompt2 = template2.invoke({'text':result1.content})
# result2 = model.invoke(prompt2)
# print(result2.content)

# Using JsonOutputParser


# prompt = template1.format()
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
chain = template1 | model | parser
result = chain.invoke({'topic': 'Langchain'})
print(result)