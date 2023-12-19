#Langchain>LECL>Interface
#https://python.langchain.com/docs/expression_language/interface    
def _chatmodel_LECL_Interface_invoke(human_input):    

    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate

    model = ChatOpenAI()

    template = """Assistant is a large language model trained by  OpenAI.

        Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

        Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

        Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

        Human: {human_input}
        Assistant:
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | model

    result = chain.invoke({"human_input":human_input})

    return result

#Langchain>LECL>Interface>Stream
#https://python.langchain.com/docs/expression_language/interface  
def _ChatModel_LECL_Stream_execute(human_input):    
    pass

#Langchain>LECL>Interface>Invoke
#https://python.langchain.com/docs/expression_language/interface  
def _ChatModel_LECL_Invoke(human_input='cat'):   
  from langchain.chat_models import ChatOpenAI
  from langchain.prompts import ChatPromptTemplate
  from langchain.schema.output_parser import StrOutputParser

  prompt = ChatPromptTemplate.from_template("tell me a joke about {foo}")
  model = ChatOpenAI()
  chain = (
      prompt 
      | model#.bind(stop=["\n"])
      | StrOutputParser()
  )
  output = chain.invoke({"foo": human_input})
  return output

#Langchain>LECL>Interface>Batch
#https://python.langchain.com/docs/expression_language/interface  
def _ChatModel_LECL_Batch_execute(human_input):    
    pass

#Langchain>LECL>Interface>Async Stream
#https://python.langchain.com/docs/expression_language/interface  
def _ChatModel_LECL_AsyncStream_execute(human_input):    
    pass

#Langchain>LECL>Interface>Async Invoke
#https://python.langchain.com/docs/expression_language/interface  
def _ChatModel_LECL_AsyncInvoke_execute(human_input):    
    pass

#Langchain>LECL>Interface>Async Batch
#https://python.langchain.com/docs/expression_language/interface  
def _ChatModel_LECL_AsyncBatch_execute(human_input):    
    pass

#Langchain>LECL>Interface>Async Stream Intermediate Steps
#https://python.langchain.com/docs/expression_language/interface  
def _ChatModel_LECL_AsyncStreamIntermediateSteps_execute(human_input):    
    pass

#Langchain>LECL>Interface>Streaming JSONPatch chunks
#https://python.langchain.com/docs/expression_language/interface  
def _ChatModel_LECL_StreamingJSONPatchChunks_execute(human_input):    
    pass

#Langchain>LECL>Interface>Streaming the incremental RunState
#https://python.langchain.com/docs/expression_language/interface  
def _ChatModel_LECL_StreamingTheIncrementalRunState_execute(human_input):    
    pass

#Langchain>LECL>Interface>Parallelism
#https://python.langchain.com/docs/expression_language/interface    
def _ChatModel_LECL_Parallelism(human_input='cat'):  

  from langchain.chat_models import ChatOpenAI
  from langchain.schema.runnable import RunnableParallel
  from langchain.prompts import ChatPromptTemplate
  from langchain.schema.output_parser import StrOutputParser

  model = ChatOpenAI()

  chain1 = ChatPromptTemplate.from_template("tell me a joke about {topic}") | model| StrOutputParser()
  chain2 = (
      ChatPromptTemplate.from_template("write a short (2 line) poem about {topic}")
      | model
      | StrOutputParser()
  )

  combined = RunnableParallel(joke=chain1, poem=chain2)

  result = combined.invoke({"topic": human_input})

  return result
    
#Langchain>LECL>Interface>Parallelism on batches
#https://python.langchain.com/docs/expression_language/interface    
def _ChatModel_LECL_ParallelismOnBatches(human_input):  
    pass

#Langchain>LECL>Interface>Vectorstore retriever
#https://python.langchain.com/docs/expression_language/interface    
#FAISS
def _ChatModel_LECL_Retriever_Chain_FAISS(human_input='James dio is dead?'):  
    
  from langchain.embeddings import OpenAIEmbeddings
  from langchain.schema.output_parser import StrOutputParser
  from langchain.schema.runnable import RunnablePassthrough
  from langchain.vectorstores import FAISS
  from langchain.chat_models import ChatOpenAI
  from langchain.prompts import ChatPromptTemplate

  texts = ["FAISS is an important library", "LangChain supports FAISS","James dio is dead"]

  vectorstore = FAISS.from_texts(
      texts, embedding=OpenAIEmbeddings()
  )
  retriever = vectorstore.as_retriever()
  template = """Answer the question based only on the following context:
  {context}

  Question: {question}
  """
  prompt = ChatPromptTemplate.from_template(template)
  model = ChatOpenAI()


  retrieval_chain = (
      {"context": retriever, "question": RunnablePassthrough()}
      | prompt
      | model
      | StrOutputParser()
  )

  '''
  vectorstore.add_texts(
      texts=["jimmy page is dead"],
      metadatas=[{"task": "damn"}],
      ids=["1111d3111"],
  )
  '''

  output = retrieval_chain.invoke(human_input)

  return output

#Chroma
def _ChatModel_LECL_Retriever_Chain_Chroma(human_input='James dio is dead?'):  
    
  from langchain.embeddings import OpenAIEmbeddings
  from langchain.schema.output_parser import StrOutputParser
  from langchain.schema.runnable import RunnablePassthrough
  from langchain.vectorstores import Chroma
  from langchain.prompts import ChatPromptTemplate
  from langchain.chat_models import ChatOpenAI
  

  texts = ["chromadb is an important library", "LangChain supports chromadb","James dio is dead"]
  vectorstore = Chroma(
      persist_directory = "./chroma_sample33334444344444", embedding_function=OpenAIEmbeddings(), 
  )
  vectorstore.add_texts(texts)
  retriever = vectorstore.as_retriever()
  template = """Answer the question based only on the following context:
  {context}

  Question: {question}
  """
  prompt = ChatPromptTemplate.from_template(template)
  model = ChatOpenAI()

  retrieval_chain = (
      {"context": retriever, "question": RunnablePassthrough()}
      | prompt
      | model
      | StrOutputParser()
  )

  '''
  vectorstore.add_texts(
      texts=["jimmy page is dead"],
      metadatas=[{"task": "damn"}],
      ids=["1111d3111"],
  )
  '''

  output = retrieval_chain.invoke(human_input)

  return output

#RAG
def _ChatModel_LECL_RAG(human_input='x raised to the third plus seven equals 12'):  
  
  from langchain.chat_models import ChatOpenAI
  from langchain.prompts import ChatPromptTemplate
  from langchain.schema import StrOutputParser
  from langchain.schema.runnable import RunnablePassthrough

  prompt = ChatPromptTemplate.from_messages(
      [
          (
              "system",
              "Write out the following equation using algebraic symbols then solve it. Use the format\n\nEQUATION:...\nSOLUTION:...\n\n",
          ),
          ("human", "{equation_statement}"),
      ]
  )
  model = ChatOpenAI(temperature=0)
  runnable = (
      {"equation_statement": RunnablePassthrough()} 
      | prompt 
      | model#.bind(stop="Therefore")
      | StrOutputParser()
  )

  output = runnable.invoke(human_input)

  return output

#LCEL Multi Chain
def _ChatModel_LECL_MultiChain(human_input):  
  from operator import itemgetter

  from langchain.chat_models import ChatOpenAI
  from langchain.prompts import ChatPromptTemplate
  from langchain.schema import StrOutputParser

  prompt1 = ChatPromptTemplate.from_template("what is the city {person} is from?")
  prompt2 = ChatPromptTemplate.from_template(
      "what country is the city {city} in? respond in {language}"
  )

  model = ChatOpenAI()

  chain1 = prompt1 | model | StrOutputParser()

  chain2 = (
      {"city": chain1, "language": itemgetter("language")}
      | prompt2
      | model
      | StrOutputParser()
  )

  #human Input 수정
  output = chain2.invoke({"person": human_input, "language": "Korean"})

  return output

#LCEL Multi Chain
def _ChatModel_LECL_MultiPromptChain(human_input="warm"):  
  from langchain.schema.runnable import RunnablePassthrough
  from langchain.chat_models import ChatOpenAI
  from langchain.prompts import ChatPromptTemplate
  from langchain.schema import StrOutputParser

  model = ChatOpenAI()

  prompt1 = ChatPromptTemplate.from_template(
      "generate a {attribute} color. Return the name of the color and nothing else:"
  )
  prompt2 = ChatPromptTemplate.from_template(
      "what is a fruit of color: {color}. Return the name of the fruit and nothing else:"
  )
  prompt3 = ChatPromptTemplate.from_template(
      "what is a country with a flag that has the color: {color}. Return the name of the country and nothing else:"
  )
  prompt4 = ChatPromptTemplate.from_template(
      "What is the color of {fruit} and the flag of {country}?"
  )

  model_parser = model | StrOutputParser()

  color_generator = (
      {"attribute": RunnablePassthrough()} | prompt1 | {"color": model_parser}
  )
  color_to_fruit = prompt2 | model_parser
  color_to_country = prompt3 | model_parser
  question_generator = (
      color_generator | {"fruit": color_to_fruit, "country": color_to_country} | prompt4
  )

  output = question_generator.invoke(human_input).messages[0].content

  return output

#SQL DB Query
#1. !apt-get update
#2. !apt-get install sqlite3
#3. !wget "https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip" -O "chinook.zip"
#4. !unzip -o -q "chinook.zip" -d "chinook_db"
def _LCEL_SQL_Query(human_input="How many employees are there?"):
  from langchain.prompts import ChatPromptTemplate

  template = """Based on the table schema below, write a SQL query that would answer the user's question:
  {schema}

  Question: {question}
  SQL Query:"""

  prompt = ChatPromptTemplate.from_template(template)

  from langchain.utilities import SQLDatabase

  db = SQLDatabase.from_uri("sqlite:////content/chinook_db/chinook.db")

  def get_schema(_):
      return db.get_table_info()

  def run_query(query):
      return db.run(query)    

  from langchain.chat_models import ChatOpenAI
  from langchain.schema.output_parser import StrOutputParser
  from langchain.schema.runnable import RunnablePassthrough

  model = ChatOpenAI()

  sql_response = (
      RunnablePassthrough.assign(schema=get_schema)
      | prompt
      | model.bind(stop=["\nSQLResult:"])
      | StrOutputParser()
  )    

  #sql_response.invoke({"question": "How many employees are there?"})

  template_response = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
  {schema}

  Question: {question}
  SQL Query: {query}
  SQL Response: {response}"""
  prompt_response = ChatPromptTemplate.from_template(template_response)

  full_chain = (
      RunnablePassthrough.assign(query=sql_response)
      | RunnablePassthrough.assign(
          schema=get_schema,
          response=lambda x: db.run(x["query"]),
      )
      | prompt_response
      | model
  )

  output = full_chain.invoke({"question": human_input}).content

  return output