from pydantic import BaseModel, Field
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import Literal, Optional
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="chat-completion",
)
model = ChatHuggingFace(llm=llm)
class Review(BaseModel):
    key_themes: list[str]= Field(description='Write down all the key themes discussed in review in a list')
    summary: str = Field(description="Brief summary of review")
    sentiment: Literal['pos','neg'] = Field(description='Give sentiment of the review')
    pros: Optional[list[str]] = Field(default=None, description='Write down all the pros inside a list')
    cons: Optional[list[str]] = Field(default=None, description='Write down all the cons inside a list')
    
structured_model = model.with_structured_output(Review)
result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy 524 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast-whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
The S-Pen integration is a great touch for note taking, although I do not use it often. Regarding camera, zooming upto 200X actually works well for distant objects, but anything beyond 30x loses quality.
However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware-why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.
Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
Cons:
Bulky and heavy-not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors""")

print(result)
