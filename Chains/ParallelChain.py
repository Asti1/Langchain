from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
load_dotenv()

prompt1 = PromptTemplate(
    template = 'Generate 5 interesting facts about {topic}',
    input_variables = ['topic']
)
prompt2 = PromptTemplate(
    template = 'Generate a quiz from the following text \n {topic}',
    input_variables = ['topic']
)
prompt3 = PromptTemplate(
    template= 'Merge the provided facts and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables = ['notes','quiz']
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

parellel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})
merge_chain = prompt3 | model1 | parser
chain = parellel_chain | merge_chain
text = """ Lesson 1: Image classification

You can click the grey arrow buttons next to the left and right panes to hide them and make more room for the video. You can search the transcript using the text box at the bottom. Scroll down this page for links to many useful resources. If you have any other suggestions for links, edits, or anything else, you'll find an "edit" link at the bottom of this (and every) notes panel.

Overview

To follow along with the lessons, you'll need to connect to a cloud GPU provider which has the fastai library installed (recommended; it should take only 5 minutes or so, and cost under $0.50/hour), or set up a computer with a suitable GPU yourself (which can take days to get working if you're not familiar with the process, so we don't recommend it). You'll also need to be familiar with the basics of the Jupyter Notebook environment we use for running deep learning experiments. Up to date tutorials and recommendations for these are available from the course website.

The key outcome of this lesson is that we'll have trained an image classifier which can recognize pet breeds at state of the art accuracy. The key to this success is the use of transfer learning, which will be a key platform for much of this course. We'll also see how to analyze the model to understand its failure modes. In this case, we'll see that the places where the model is making mistakes is in the same areas that even breeding experts can make mistakes.

We'll discuss the overall approach of the course, which is somewhat unusual in being top-down rather than bottom-up. So rather than starting with theory, and only getting to practical applications later, instead we start with practical applications, and then gradually dig deeper and deeper in to them, learning the theory as needed. This approach takes more work for teachers to develop, but it's been shown to help students a lot, for example in education research at Harvard by David Perkins.

We also discuss how to set the most important hyper-parameter when training neural networks: the learning rate, using Leslie Smith's fantastic learning rate finder method. Finally, we'll look at the important but rarely discussed topic of labeling, and learn about some of the features that fastai provides for allowing you to easily add labels to your images."""
result = chain.invoke({'topic': text})
print(result)
chain.get_graph().print_ascii()