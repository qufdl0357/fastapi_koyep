import os
from dotenv import load_dotenv
from typing import Union
from fastapi import FastAPI
from app.routes import router as api_router

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
GOOGLE_CSE_ID  = os.environ.get('GOOGLE_CSE_ID')

app = FastAPI()

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

    