#Langchain>Modules>Retrieval>Document Loader>TextLoader
#https://python.langchain.com/docs/modules/data_connection/document_loaders/
"""
************************************************************************************REVISE
"""
def _Modules_TextLoader_Upsert(human_input):
  from langchain.embeddings.openai import OpenAIEmbeddings
  from langchain.chat_models.openai import ChatOpenAI
  from langchain.document_loaders import TextLoader
  from langchain.vectorstores import Chroma
  from langchain.chains import RetrievalQA
  from langchain.text_splitter import CharacterTextSplitter

  llm = ChatOpenAI()

  embeddings = OpenAIEmbeddings()

  loader = TextLoader("app/sample_data/meow.txt")

  data = loader.load()

  text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
  texts = text_splitter.split_documents(data)

  db = Chroma.from_documents(texts, embeddings, persist_directory = "./chroma_db_apify2")

  # expose this index in a retriever interface
  retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
  # create a chain to answer questions
  qa = RetrievalQA.from_chain_type(
      llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

  result = qa({"query": human_input})

  return result

#Langchain>Modules>Retrieval>Document Loader>TextLoader
#https://python.langchain.com/docs/modules/data_connection/document_loaders/
def _Modules_TextLoader_Query(human_input="I need to know about california_housing"):
  from langchain.embeddings.openai import OpenAIEmbeddings
  from langchain.chat_models.openai import ChatOpenAI
  from langchain.document_loaders import TextLoader
  from langchain.vectorstores import Chroma
  from langchain.chains import RetrievalQA
  from langchain.text_splitter import CharacterTextSplitter

  llm = ChatOpenAI()
  embeddings = OpenAIEmbeddings()

  db = Chroma(
    embedding_function = embeddings, persist_directory = "./chroma_db_apify2"
  )

  # expose this index in a retriever interface
  retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
  # create a chain to answer questions
  qa = RetrievalQA.from_chain_type(
      llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

  result = qa({"query": human_input})

  return result

#Langchain>Modules>Retrieval>Document Loader>CSV Loader
#https://python.langchain.com/docs/modules/data_connection/document_loaders/csv
def _Modules_CSVLoader(human_input="avg of total_rooms"):
  from langchain.document_loaders.csv_loader import CSVLoader
  from langchain.chains import create_sql_query_chain

  llm = ChatOpenAI()

  loader = CSVLoader(file_path='./sample_data/california_housing_test.csv')
  data = loader.load()

  return data

#Langchain>Modules>Retrieval>Document Loader>CSV Loader
#https://python.langchain.com/docs/modules/data_connection/document_loaders/csv
def _Modules_CSVAgent(human_input="avg of total_rooms"):

  from langchain.agents.agent_types import AgentType
  from langchain.chat_models import ChatOpenAI
  from langchain.llms import OpenAI
  from langchain_experimental.agents.agent_toolkits import create_csv_agent

  llm = ChatOpenAI()

  agent = create_csv_agent(
      llm,
      "./sample_data/california_housing_test.csv",
      verbose=True,
      agent_type=AgentType.OPENAI_FUNCTIONS,
  )

  output = agent.run(human_input)

  return output

#Langchain>Modules>Retrieval>Document Loader>Directory Loader
#https://python.langchain.com/docs/modules/data_connection/document_loaders/file_directory
def _Modules_DirectoryLoader(human_input="what is this about"):

  from langchain.document_loaders import DirectoryLoader
  from langchain.embeddings.openai import OpenAIEmbeddings
  from langchain.chat_models.openai import ChatOpenAI
  from langchain.document_loaders import TextLoader
  from langchain.vectorstores import Chroma
  from langchain.chains import RetrievalQA
  from langchain.text_splitter import CharacterTextSplitter

  llm = ChatOpenAI()

  embeddings = OpenAIEmbeddings()

  loader = DirectoryLoader(
      './sample_data/',
      glob="**/*.md",
      loader_cls=TextLoader, #TextLoader/PythonLoader
      silent_errors=True,
      #show_progress=True
  )

  docs = loader.load()

  doc_sources = [doc.metadata['source']  for doc in docs]

  print(doc_sources)

  text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
  texts = text_splitter.split_documents(docs)

  db = Chroma.from_documents(texts, embeddings, persist_directory = "./chroma_db_1")

  # expose this index in a retriever interface
  retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
  # create a chain to answer questions
  qa = RetrievalQA.from_chain_type(
      llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

  result = qa({"query": human_input})

  return result

#Langchain>Modules>Retrieval>Document Loader>HTML
#https://python.langchain.com/docs/modules/data_connection/document_loaders/html
def _Modules_HTMLLoader(human_input="what is this about"):
  pass

#Langchain>Modules>Retrieval>Document Loader>JSON
#https://python.langchain.com/docs/modules/data_connection/document_loaders/json
def _Modules_JSONLoader(human_input="what is this about"):
  pass

#Langchain>Modules>Retrieval>Document Loader>Markdown
#https://python.langchain.com/docs/modules/data_connection/document_loaders/json
def _Modules_MarkdownLoader(human_input="what is this about"):
  pass

#Langchain>Modules>Retrieval>Document Loader>PDF>PyPDF
#Langchain>Modules>Retrieval>Document Transformer>RecursiveCharacterTextSplitter
#https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf
#https://python.langchain.com/docs/modules/data_connection/document_transformers/
#!pip install PyPDF
def _Modules_PyPDFLoader(human_input="what is this about"):
  
  from langchain.document_loaders import PyPDFLoader
  from langchain.embeddings.openai import OpenAIEmbeddings
  from langchain.chat_models.openai import ChatOpenAI
  from langchain.vectorstores import Chroma
  from langchain.text_splitter import RecursiveCharacterTextSplitter

  llm = ChatOpenAI()

  loader = PyPDFLoader("./(주)원익큐엔씨_상세기업정보보고서.pdf")
  pages = loader.load()
  print(pages[0].metadata)

  text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
    add_start_index = True,
  )

  texts = text_splitter.split_documents(pages)

  embeddings = OpenAIEmbeddings()

  # Chroma DB 에 저장
  db = Chroma.from_documents(texts,embeddings,persist_directory="./chroma_db3_apify")

  # 검증: retriever 가져옴
  retriever = db.as_retriever()

  # langchain hub 에서 Prompt 다운로드 예시
  # https://smith.langchain.com/hub/rlm/rag-prompt

  from langchain import hub

  rag_prompt = hub.pull("rlm/rag-prompt")
  print(rag_prompt)

  # RAG chain 생성
  from langchain.schema.runnable import RunnablePassthrough

  # pipe operator를 활용한 체인 생성
  rag_chain = (
      {"context": retriever, "question": RunnablePassthrough()}
      | rag_prompt
      | llm
  )

  output = rag_chain.invoke(human_input)

  return output

#Langchain>Modules>Retrieval>Document Loader>PDF>PyMuPDF
#Langchain>Modules>Retrieval>Document Transformer>RecursiveCharacterTextSplitter
#https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf
#https://python.langchain.com/docs/modules/data_connection/document_transformers/
#!pip install PyMuPDF
def _Modules_PyMuPDFLoader(human_input="what is this about"):
  
  from langchain.document_loaders import PyPDFLoader
  from langchain.embeddings.openai import OpenAIEmbeddings
  from langchain.chat_models.openai import ChatOpenAI
  from langchain.vectorstores import Chroma
  from langchain.text_splitter import RecursiveCharacterTextSplitter

  llm = ChatOpenAI()
  from langchain.document_loaders import PyMuPDFLoader


  loader = PyMuPDFLoader("./(주)원익큐엔씨_상세기업정보보고서.pdf")

  data = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
    add_start_index = True,
  )

  texts = text_splitter.split_documents(data)

  print(texts[0].metadata)
  embeddings = OpenAIEmbeddings()

  # Chroma DB 에 저장
  db = Chroma.from_documents(texts,embeddings,persist_directory="./chroma_db3_apify")

  # 검증: retriever 가져옴
  retriever = db.as_retriever()

  # langchain hub 에서 Prompt 다운로드 예시
  # https://smith.langchain.com/hub/rlm/rag-prompt

  from langchain import hub

  rag_prompt = hub.pull("rlm/rag-prompt")
  print(rag_prompt)

  # RAG chain 생성
  from langchain.schema.runnable import RunnablePassthrough

  # pipe operator를 활용한 체인 생성
  rag_chain = (
      {"context": retriever, "question": RunnablePassthrough()}
      | rag_prompt
      | llm
  )

  output = rag_chain.invoke(human_input)

  return output

#Langchain>Modules>Retrieval>Document Loader>PDF>PyPDF Directory
#Langchain>Modules>Retrieval>Document Transformer>RecursiveCharacterTextSplitter
#https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf
#https://python.langchain.com/docs/modules/data_connection/document_transformers/
def _Modules_PyPDFDirectoryLoader(human_input="what is this about"):
  
  from langchain.document_loaders import PyPDFLoader
  from langchain.embeddings.openai import OpenAIEmbeddings
  from langchain.chat_models.openai import ChatOpenAI
  from langchain.vectorstores import Chroma
  from langchain.document_loaders import PyPDFDirectoryLoader
  from langchain.text_splitter import RecursiveCharacterTextSplitter

  llm = ChatOpenAI()
  
  loader = PyPDFDirectoryLoader("./")

  data = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(
      # Set a really small chunk size, just to show.
      chunk_size = 1000,
      chunk_overlap  = 200,
      length_function = len,
      add_start_index = True,
    )
  texts = text_splitter.split_documents(data)

  print(texts[0].metadata)
  embeddings = OpenAIEmbeddings()

  # Chroma DB 에 저장
  db = Chroma.from_documents(texts,embeddings,persist_directory="./chroma_db3_apify")

  # 검증: retriever 가져옴
  retriever = db.as_retriever()

  # langchain hub 에서 Prompt 다운로드 예시
  # https://smith.langchain.com/hub/rlm/rag-prompt

  from langchain import hub

  rag_prompt = hub.pull("rlm/rag-prompt")
  print(rag_prompt)

  # RAG chain 생성
  from langchain.schema.runnable import RunnablePassthrough

  # pipe operator를 활용한 체인 생성
  rag_chain = (
      {"context": retriever, "question": RunnablePassthrough()}
      | rag_prompt
      | llm
  )

  output = rag_chain.invoke(human_input)

  return output