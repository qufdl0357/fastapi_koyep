from fastapi import APIRouter
from app.model import Input, Output

from app.utils._Chatting_Model import (
    _ChattingModel_Invoke
)


router = APIRouter()

@router.post("/chatInvoke", response_model=Output)
async def chattingModel_Invoke(input: Input):
    output = _ChattingModel_Invoke(human_input=input.human_input)
    return {"output":output}