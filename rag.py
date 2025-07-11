import time

start_time = time.time()
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

class CompanyInfo(BaseModel):
    name: Optional[Union[str, List[str]]] = Field(default=None,description="Company name")
    industry: Optional[Union[str, List[str]]] = Field(default=None,description="Company industry")
    headquarters: Optional[Union[str, List[str]]] = Field(default=None,description="Headquarters location")
    company_size: Optional[Union[str, List[str]]] = Field(default=None,description="Number of employees")
    website:Optional[Union[str,List[str]]]=Field(default=None,description="Give the website of the company")
    country:Optional[Union[str,List[str]]]=Field(default=None,description="Give the country of the company")

    phone_number:Optional[Union[str,List[str]]]=Field(default=None,description="Give the phone number of company if  available")
    email=Optional[Union[str,List[str]]]=Field(default=None,description="Give the email of the company if mentioned")
    revenue:Optional[Union[str,List[str]]]=Field(default=None,description="Give the revenue of the company if mentioned")









company_list2 = []###list of scraped companies information
company_list1 = []
# Example Knowledge Base


import numpy as np
import faiss
res=[]
for i in company_list2:
    query=i
    knowledge_base=[i]


        # Vectorize Knowledge Base
    model = SentenceTransformer('all-MiniLM-L6-v2')






    kb_embeddings = model.encode(knowledge_base)

        # Build FAISS Index
    index = faiss.IndexFlatL2(kb_embeddings.shape[1])
    index.add(np.array(kb_embeddings))

        # Convert Query to Vector
    query_embedding = model.encode([query])

        # Retrieve Top 2 Relevant Documents
    D, I = index.search(query_embedding, k=2)
    retrieved_docs = [knowledge_base[i] for i in I[0]]

    print("Retrieved Documents:", retrieved_docs)
    llm = OllamaLLM(model="deepseek-r1:7b")
    parser = PydanticOutputParser(pydantic_object=CompanyInfo)
    prompt = PromptTemplate(
                template="""
                    Extract company information from the following text:
                    Original Text: {original_text}
                    Retrieved Data: {retrieved_data}
                
                    - The output MUST be a valid JSON following this format:
                    {{
                        "name": "Company Name",
                        "industry": "Industry of company",
                        "headquarters": "Headquarters Location",
                        "company_size": "Number of Employees",
                        "website": "Company Website URL",
                        "country": "Country in which company is located",
                        "phone_number": "Company Phone Number (if available, otherwise empty string)"
                        "email":"Company email(if available, otherwise empty string)
                        "revenue": "Revenue of company (if available, otherwise empty string)"
                        
                    }}
                    
                    - Ensure all fields are always present.
                    - If a field is missing, use an empty string `""` instead of `null` or omitting it.
                
                    {format_instructions}
    
                """,
                input_variables=["original_text","retrieved_data"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
                )

        # Chain
    chain = prompt | llm | parser

        # Final Extraction
    result = chain.invoke({
                    "original_text": query,
                    "retrieved_data": " ".join(retrieved_docs)
                })

    print(result)
end_time = time.time()  # Record end time

execution_time = int(end_time - start_time)  # Convert to whole number seconds
print(execution_time)


