def _Modules_TextLoader_Query(human_input):
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