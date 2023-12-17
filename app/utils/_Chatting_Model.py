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