from fastapi import APIRouter
from app.model import Input, Output, Keyword
from app.utils._LangChain_GetStarted import (
    _LLM_execute,
    _ChatModel_execute,
    _ChatModel_LLMChain_PromptTemplate_execute,
    _ChatModel_OutputParser_LECL_execute
)

from app.utils._LECL_Cookbook import (
    _chatmodel_LECL_Interface_invoke,
    _ChatModel_LECL_Invoke,
    _ChatModel_LECL_Parallelism,
    #_ChatModel_LECL_Retriever_Chain,
    _ChatModel_LECL_Retriever_Chain_Chroma,
    _ChatModel_LECL_RAG,
    _ChatModel_LECL_MultiChain,
    _ChatModel_LECL_MultiPromptChain,
    _LCEL_SQL_Query
)

from app.utils._Prompt_Templates import (
    _Modules_PromptTemplate,
    _Modules_PromptTemplate_input_variables,
    _Modules_ChatPromptTemplate,
    _Modules_FewShotPromptTemplate,
    _Modules_ExampleSelector,
    _Modules_FewShotChatModelPromptTemplate,
    _Modules_DynamicFewShotPrompt
)

from app.utils._Chatting_Model import (
    _ChattingModel_Invoke,
    _Modules_MessagePromptTemplate,
    _Modules_Chat_PromptTemplate,
    _Modules_TrackingTokenUsage
)

from app.utils._Document_Loader import (
    _Modules_TextLoader_Upsert,
    _Modules_TextLoader_Query,
    _Modules_CSVLoader,
    _Modules_CSVAgent,
    _Modules_DirectoryLoader,
    _Modules_PyPDFLoader,
    _Modules_PyMuPDFLoader,
    _Modules_PyPDFDirectoryLoader
)

from app.utils._Retriever import (
    _Modules_MultiQueryRetriever
)

from app.utils._Vectara_Apify_Crawling import (
    _Insert_Vectara
)

router = APIRouter()

#_Prompt_Templates.py
@router.post("/_modules_PromptTemplate/", response_model=Output)
async def modules_PromptTemplate(input: Input):
    llm_output = _Modules_PromptTemplate(adjective=input.adjective,content=input.content)
    return {"output": llm_output}

@router.post("/_modules_PromptTemplate_input_variables/", response_model=Output)
async def modules_PromptTemplate_input_variables(input: Input):
    llm_output = _Modules_PromptTemplate_input_variables(adjective=input.adjective,content=input.content)
    return {"output": llm_output}

@router.post("/_modules_ChatPromptTemplate/", response_model=Output)
async def modules_ChatPromptTemplate(input: Input):
    llm_output = _Modules_ChatPromptTemplate(human_input=input.human_input)
    return {"output": llm_output}

@router.post("/_modules_FewShotPromptTemplate/", response_model=Output)
async def modules_FewShotPromptTemplate(input: Input):
    llm_output = _Modules_FewShotPromptTemplate(human_input=input.human_input)
    return {"output": llm_output}

@router.post("/_modules_ExampleSelector/", response_model=Output)
async def modules_ExampleSelector(input: Input):
    llm_output = _Modules_ExampleSelector(human_input=input.human_input)
    return {"output": llm_output}

@router.post("/_modules_FewShotChatModelPromptTemplate/", response_model=Output)
async def modules_FewShotChatModelPromptTemplate(input: Input):
    llm_output = _Modules_FewShotChatModelPromptTemplate(human_input=input.human_input)
    return {"output": llm_output}

@router.post("/_modules_DynamicFewShotPrompt/", response_model=Output)
async def modules_DynamicFewShotPrompt(input: Input):
    llm_output = _Modules_DynamicFewShotPrompt(human_input=input.human_input)
    return {"output": llm_output}

#_Chatting_Model.py
@router.post("/_chattingModel_Invoke/", response_model=Output)
async def chattingModel_Invoke(input: Input):
    llm_output = _ChattingModel_Invoke(human_input=input.human_input)
    return {"output": llm_output}

@router.post("/_modules_MessagePromptTemplate/", response_model=Output)
async def modules_MessagePromptTemplate(input: Input):
    llm_output = _Modules_MessagePromptTemplate(input_language=input.input_language,output_language=input.output_language,text=input.text)
    return {"output": llm_output}

@router.post("/_modules_Chat_PromptTemplate/", response_model=Output)
async def modules_PromptTemplate(input: Input):
    llm_output = _Modules_Chat_PromptTemplate(input_language=input.input_language,output_language=input.output_language,text=input.text)
    return {"output": llm_output}

@router.post("/_modules_TrackingTokenUsage/", response_model=Output)
async def modules_TrackingTokenUsage(input: Input):
    llm_output = _Modules_TrackingTokenUsage(human_input=input.human_input)
    return {"output": llm_output}

#_Document_Loader.py
@router.post("/_modules_TextLoader_Upsert/", response_model=Output)
async def modules_TextLoader_Upsert(input: Input):
    llm_output = _Modules_TextLoader_Upsert(human_input=input.human_input)
    return {"output": llm_output}

@router.post("/_modules_TextLoader_Query/", response_model=Output)
async def modules_TextLoader_Query(input: Input):
    llm_output = _Modules_TextLoader_Query(human_input=input.human_input)
    return {"output": llm_output}

@router.post("/_modules_CSVLoader/", response_model=Output)
async def modules_CSVLoader(input: Input):
    llm_output = _Modules_CSVLoader(human_input=input.human_input)
    return {"output": llm_output}

@router.post("/_modules_CSVAgent/", response_model=Output)
async def modules_CSVAgent(input: Input):
    llm_output = _Modules_CSVAgent(human_input=input.human_input)
    return {"output": llm_output}

@router.post("/_modules_DirectoryLoader/", response_model=Output)
async def modules_DirectoryLoader(input: Input):
    llm_output = _Modules_DirectoryLoader(human_input=input.human_input)
    return {"output": llm_output}

@router.post("/_modules_PyPDFLoader/", response_model=Output)
async def modules_PyPDFLoader(input: Input):
    llm_output = _Modules_PyPDFLoader(human_input=input.human_input)
    return {"output": llm_output}

@router.post("/_modules_PyMuPDFLoader/", response_model=Output)
async def modules_PyMuPDFLoader(input: Input):
    llm_output = _Modules_PyMuPDFLoader(human_input=input.human_input)
    return {"output": llm_output}

@router.post("/_modules_PyPDFDirectoryLoader/", response_model=Output)
async def modules_PyPDFDirectoryLoader(input: Input):
    llm_output = _Modules_PyPDFDirectoryLoader(human_input=input.human_input)
    return {"output": llm_output}

#_Retriever.py
@router.post("/_modules_MultiQueryRetriever/", response_model=Output)
async def modules_MultiQueryRetriever(input: Input):
    llm_output = _Modules_MultiQueryRetriever(human_input=input.human_input)
    return {"output": llm_output}

#_Vectara_Apify_Crawling.py
@router.post("/_insert_Vectara", response_model=Output)
async def insert_Vectara(input: Keyword):
    llm_output = _Insert_Vectara(keyword=input.keyword)
    return {"output": llm_output}