#Langchain>Get started>Quickstart>LLM
#https://python.langchain.com/docs/get_started/quickstart
def _LLM_execute(human_input):

    from langchain.llms import OpenAI
    from langchain.schema import HumanMessage

    llm = OpenAI()
    messages = [HumanMessage(content=human_input)]

    output = llm.invoke(messages)

    return output

#Langchain>Get started>Quickstart>Chat Model
#https://python.langchain.com/docs/get_started/quickstart
def _ChatModel_execute(human_input):

    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage

    chat_model = ChatOpenAI()
    messages  = [HumanMessage(content=human_input)]

    messages_to_str = chat_model.invoke(messages)
    output = messages_to_str.content
    #output = chat_model.invoke(messages)

    return output

#Langchain>Get started>Quickstart>Prompt templates
#https://python.langchain.com/docs/get_started/quickstart

#Langchain>Modules>More>Chains>LLM
#https://python.langchain.com/docs/modules/chains/foundational/llm_chain
def _ChatModel_LLMChain_PromptTemplate_execute(human_input):

    from langchain.chains import LLMChain
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate

    chat_model = ChatOpenAI()
    template = """Assistant is a large language model trained by  OpenAI.

        Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

        Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

        Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

        Human: {human_input}
        Assistant:
    """

    prompt = PromptTemplate.from_template(template)

    llm_chain  = LLMChain(
            llm=chat_model,
            prompt=prompt,
            verbose=True,
        )

    output = llm_chain.predict (human_input=human_input)
    return output

#Langchain>Get started>Quickstart>Output parsers
#Langchain>Get started>Quickstart>Composing with LCEL
#https://python.langchain.com/docs/get_started/quickstart    
def _ChatModel_OutputParser_LECL_execute(human_input):    
    
    from typing import List

    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema import BaseOutputParser

    class CommaSeparatedListOutputParser(BaseOutputParser):
        """Parse the output of an LLM call to a comma-separated list."""

        def parse(self, text: str):
            """Parse the output of an LLM call."""
            return text.strip().split(", ")
        
    template = """You are a helpful assistant who generates comma separated lists.
        A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
        ONLY return a comma separated list, and nothing more."""
    
    human_template = "{text}"

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])

    chain = chat_prompt | ChatOpenAI() | CommaSeparatedListOutputParser()

    result = chain.invoke({"text": human_input})

    return result
