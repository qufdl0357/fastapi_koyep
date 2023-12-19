from pydantic import BaseModel
#from utils import generate_description
#from utils import conversation

class Input(BaseModel):
    human_input: str
    adjective: str
    content: str
    input_language:str
    output_language:str
    text:str

class Output(BaseModel):
    output: str | list | dict

class Keyword(BaseModel):
    keyword: str