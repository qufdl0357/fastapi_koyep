from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

#Langchain>Modules>Model I/O>Chatting Model
#https://python.langchain.com/docs/modules/model_io/chat/
def _ChattingModel_Invoke(human_input):
    from langchain.chat_models import ChatOpenAI

    chat = ChatOpenAI()

    from langchain.schema.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content="You're a helpful assistant"),
        HumanMessage(content=human_input),
    ]

    output = chat.invoke(messages)
    return output

#Langchain>Modules>Model I/O>Chatting Model>Caching
#https://python.langchain.com/docs/modules/model_io/chat/
from langchain.globals import set_llm_cache
#from langchain_community.cache import InMemoryCache
from langchain.chat_models import ChatOpenAI


#set_llm_cache(InMemoryCache())

def _ChattingModel_Caching(chat, human_input):
    
    from langchain.schema.messages import HumanMessage, SystemMessage

    chat = ChatOpenAI()
    messages = [
        SystemMessage(content="You're a helpful assistant"),
        HumanMessage(content=human_input),
    ]

    output = chat.invoke(messages).content
    return output

#Langchain>Modules>Model I/O>ChatModel>Caching>In Memory Cache
#https://python.langchain.com/docs/modules/model_io/chat/chat_model_caching
from langchain.globals import set_llm_cache
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

#from langchain.cache import InMemoryCache
#set_llm_cache(InMemoryCache())

def _Modules_MemoryCache(human_input):
  output = llm.predict(human_input)
  return output

#Langchain>Modules>Model I/O>ChatModel>Caching>SQLite Cache
#https://python.langchain.com/docs/modules/model_io/chat/chat_model_caching
#1. !apt-get update
#2. !apt-get install sqlite3
from langchain.globals import set_llm_cache
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI()

# We can do the same thing with a SQLite cache
#from langchain.cache import SQLiteCache
#set_llm_cache(SQLiteCache(database_path=".langchain.db"))

def _Modules_MemoryCache(human_input):
  output = llm.predict(human_input)
  return output

#Langchain>Modules>Model I/O>ChatModel>Prompt>Message Prompt Template
#https://python.langchain.com/docs/modules/model_io/chat/prompts
def _Modules_MessagePromptTemplate(input_language, output_language, text):

  from langchain.chat_models import ChatOpenAI
  from langchain.prompts import PromptTemplate
  from langchain.prompts.chat import (
      ChatPromptTemplate,
      SystemMessagePromptTemplate,
      AIMessagePromptTemplate,
      HumanMessagePromptTemplate,
  )

  llm = ChatOpenAI()

  system_template="You are a helpful assistant that translates {input_language} to {output_language}."
  system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

  human_template="{text}"
  human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

  chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

  prompt = chat_prompt.format_prompt(input_language=input_language, output_language=output_language, text=text).to_messages()

  result = llm.invoke(prompt).content

  return result

#Langchain>Modules>Model I/O>ChatModel>Prompt>Prompt Template
#https://python.langchain.com/docs/modules/model_io/chat/prompts
def _Modules_Chat_PromptTemplate(input_language, output_language, text):

  from langchain.chat_models import ChatOpenAI
  from langchain.prompts import PromptTemplate
  from langchain.prompts.chat import (
      ChatPromptTemplate,
      SystemMessagePromptTemplate,
      AIMessagePromptTemplate,
      HumanMessagePromptTemplate,
  )

  llm = ChatOpenAI()

  system_template="You are a helpful assistant that translates {input_language} to {output_language}."
  system_prompt = PromptTemplate(
      template=system_template,
      input_variables=["input_language","output_language"]
  )

  system_message_prompt = SystemMessagePromptTemplate(prompt=system_prompt)

  human_template="{text}"
  human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

  chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

  prompt = chat_prompt.format_prompt(input_language=input_language, output_language=output_language, text=text).to_messages()

  result = llm.invoke(prompt).content

  return result

#Langchain>Modules>Model I/O>ChatModel>Tracking token usage
#https://python.langchain.com/docs/modules/model_io/chat/token_usage_tracking
def _Modules_TrackingTokenUsage(human_input):

  from langchain.callbacks import get_openai_callback
  from langchain.chat_models import ChatOpenAI

  llm = ChatOpenAI()

  with get_openai_callback() as cb:
    result = llm.invoke(human_input)
    print(cb)

  return result.content

