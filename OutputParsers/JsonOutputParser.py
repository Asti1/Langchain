from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)
parser = JsonOutputParser()
template1 = PromptTemplate(
    template= "Give me the name, age and city of a fictional person \n {format_instruction}",
    input_variables = [],
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
result = chain.invoke({})
print(result)