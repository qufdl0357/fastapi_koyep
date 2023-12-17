from fastapi import FastAPI
from .routes import router

app = FastAPI()


@app.get("/")
def read_root():
    return {"FastAPI Test": "Langchain Test"}


app.include_router(router, prefix='/langchain')