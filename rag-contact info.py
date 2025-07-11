
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_ollama import OllamaLLM
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, HttpUrl
from langchain.prompts import PromptTemplate


from typing import Union, List, Optional
from pydantic import BaseModel, Field, validator

from typing import Union, List
from pydantic import BaseModel, Field

class ContactInfo(BaseModel):
    name:str=Field(default=None,description="Name of the person")
    city:str=Field(default=None,description="Give the city name")
    state:str=Field(default=None,description="Give the state name")
    country:str=Field(default=None,description="Give the country name")
    company:str=Field(default=None,description="Name of the company")
    job_title:str=Field(default=None,description="Give the job title")
    joining:str=Field(default=None,description="Give the month and year of joining current company")
    experience:str=Field(default=None,description="Give the total experience across all companies")
    birth_date:str=Field(default=None,description="Give the birth date i mentioned")
    email:str=Field(default=None,description="Give the email if available")
    phone_number:str=Field(default=None,description="Give the phone number if available")


llm = OllamaLLM(model="deepseek-r1:7b")
parser = PydanticOutputParser(pydantic_object=ContactInfo)
text_list=[]
format_instructions=parser.get_format_instructions()
prompt = PromptTemplate(
    template="""
                    Extract the information from the following text:
                    {original_text}

                    - The output MUST be a valid JSON following this format:
                    {{
                        "name": "Name of the person",
                        "company: "Company name",
                        "city": "Give the city name",
                        "state":"Give the state name",
                        "country":"Give the country",
                        "Job title":"Give the job title",
                        "Joining::"Give the month or year of joining the current company",
                        "Experience":"Give the total experience across all companies",
                        "Birth date":"Give the birth date if mentioned",
                        "email": "Give the email (if available, otherwise empty string)"
                        "phone number": "Give the phone number (if available, otherwise empty string)"

                    }}

                    - Ensure all fields are always present.
                    - If a field is missing, use an empty string `""` instead of `null` or omitting it.

                    {format_instructions}

                """,
    input_variables=["original_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
chain = prompt | llm | parser

for i in text_list:
    result = chain.invoke({
                        "original_text": i,
                        "retrieved_data": " ".join(retrieved_docs)
                    })
    print(result)
