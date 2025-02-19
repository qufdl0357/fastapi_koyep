import os
from dotenv import load_dotenv
from typing import Union
from fastapi import FastAPI
from app.routes import router as api_router

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
GOOGLE_CSE_ID  = os.environ.get('GOOGLE_CSE_ID')

app = FastAPI()
app.include_router(api_router)

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, ws_ping_timeout=600)

