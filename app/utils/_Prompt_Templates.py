#Langchain>Modules>Model I/O>Prompt>PromptTemplate
#https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/   
def _Modules_PromptTemplate(adjective, content):    

    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate

    llm = OpenAI()

    prompt_template = PromptTemplate.from_template("Tell me a {adjective} joke about {content}.")
    prompt = prompt_template.format(adjective=adjective, content=content)

    result = llm(prompt)

    return result

#Langchain>Modules>Model I/O>Prompt>PromptTemplate
#https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/   
def _Modules_PromptTemplate_input_variables(adjective, content):    

    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate

    llm = OpenAI()

    prompt_template = PromptTemplate(
      input_variables=["adjective","content"],
      template="Tell me a {adjective} joke about {content}.",
    )

    prompt = prompt_template.format(adjective=adjective, content=content)

    result = llm(prompt)

    return result

#Langchain>Modules>Model I/O>Prompt>ChatPromptTemplate
#https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/   
def _Modules_ChatPromptTemplate(human_input):    

    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.prompts import HumanMessagePromptTemplate
    from langchain.schema.messages import SystemMessage

    llm = ChatOpenAI()

    chat_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            content=(
                "You are a helpful assistant that re-writes the user's text to "
                "sound more upbeat."
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ])

    prompt = chat_template.format_messages(text=human_input)

    result = llm(prompt)

    return result

#Langchain>Modules>Model I/O>Prompt>PromptTemplate>CustomPromptTemplate
#https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/custom_prompt_template
def _Modules_CustomPromptTemplate(human_input):    
    pass

#Langchain>Modules>Model I/O>Prompt>PromptTemplate>FewShotPromptTemplate
#https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/few_shot_examples
def _Modules_FewShotPromptTemplate(human_input):    
    
    from langchain.prompts.few_shot import FewShotPromptTemplate
    from langchain.prompts.prompt import PromptTemplate
    from langchain.llms import OpenAI

    examples = [
    {
        "question": "Who lived longer, Muhammad Ali or Alan Turing?",
        "answer":
        """
        Are follow up questions needed here: Yes.
        Follow up: How old was Muhammad Ali when he died?
        Intermediate answer: Muhammad Ali was 74 years old when he died.
        Follow up: How old was Alan Turing when he died?
        Intermediate answer: Alan Turing was 41 years old when he died.
        So the final answer is: Muhammad Ali
        """
    },
    {
        "question": "When was the founder of craigslist born?",
        "answer":
        """
        Are follow up questions needed here: Yes.
        Follow up: Who was the founder of craigslist?
        Intermediate answer: Craigslist was founded by Craig Newmark.
        Follow up: When was Craig Newmark born?
        Intermediate answer: Craig Newmark was born on December 6, 1952.
        So the final answer is: December 6, 1952
        """
    },
    {
        "question": "Who was the maternal grandfather of George Washington?",
        "answer":
        """
        Are follow up questions needed here: Yes.
        Follow up: Who was the mother of George Washington?
        Intermediate answer: The mother of George Washington was Mary Ball Washington.
        Follow up: Who was the father of Mary Ball Washington?
        Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
        So the final answer is: Joseph Ball
        """
    },
    {
        "question": "Are both the directors of Jaws and Casino Royale from the same country?",
        "answer":
        """
        Are follow up questions needed here: Yes.
        Follow up: Who is the director of Jaws?
        Intermediate Answer: The director of Jaws is Steven Spielberg.
        Follow up: Where is Steven Spielberg from?
        Intermediate Answer: The United States.
        Follow up: Who is the director of Casino Royale?
        Intermediate Answer: The director of Casino Royale is Martin Campbell.
        Follow up: Where is Martin Campbell from?
        Intermediate Answer: New Zealand.
        So the final answer is: No
        """
    }
    ]
    
    example_prompt = PromptTemplate(input_variables=["question", "answer"], template="Question: {question}\n{answer}")

    template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix="Question: {input}",
        input_variables=["input"]
    )

    prompt = template.format(input="Who was the father of Mary Ball Washington?")

    llm = OpenAI()

    output = llm.invoke(prompt)    

    return output

#Langchain>Modules>Model I/O>Prompt>PromptTemplate>ExampleSelector
#https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/few_shot_examples
def _Modules_ExampleSelector(human_input):    
    
    from langchain.prompts.few_shot import FewShotPromptTemplate
    from langchain.prompts.prompt import PromptTemplate
    from langchain.llms import OpenAI
    from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
    from langchain.vectorstores import Chroma
    from langchain.embeddings import OpenAIEmbeddings

    examples = [
    {
        "question": "Who lived longer, Muhammad Ali or Alan Turing?",
        "answer":
        """
        Are follow up questions needed here: Yes.
        Follow up: How old was Muhammad Ali when he died?
        Intermediate answer: Muhammad Ali was 74 years old when he died.
        Follow up: How old was Alan Turing when he died?
        Intermediate answer: Alan Turing was 41 years old when he died.
        So the final answer is: Muhammad Ali
        """
    },
    {
        "question": "When was the founder of craigslist born?",
        "answer":
        """
        Are follow up questions needed here: Yes.
        Follow up: Who was the founder of craigslist?
        Intermediate answer: Craigslist was founded by Craig Newmark.
        Follow up: When was Craig Newmark born?
        Intermediate answer: Craig Newmark was born on December 6, 1952.
        So the final answer is: December 6, 1952
        """
    },
    {
        "question": "Who was the maternal grandfather of George Washington?",
        "answer":
        """
        Are follow up questions needed here: Yes.
        Follow up: Who was the mother of George Washington?
        Intermediate answer: The mother of George Washington was Mary Ball Washington.
        Follow up: Who was the father of Mary Ball Washington?
        Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
        So the final answer is: Joseph Ball
        """
    },
    {
        "question": "Are both the directors of Jaws and Casino Royale from the same country?",
        "answer":
        """
        Are follow up questions needed here: Yes.
        Follow up: Who is the director of Jaws?
        Intermediate Answer: The director of Jaws is Steven Spielberg.
        Follow up: Where is Steven Spielberg from?
        Intermediate Answer: The United States.
        Follow up: Who is the director of Casino Royale?
        Intermediate Answer: The director of Casino Royale is Martin Campbell.
        Follow up: Where is Martin Campbell from?
        Intermediate Answer: New Zealand.
        So the final answer is: No
        """
    }
    ]

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        # This is the list of examples available to select from.
        examples,
        # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
        OpenAIEmbeddings(),
        # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
        Chroma,
        # This is the number of examples to produce.
        k=1
    )
    
    example_prompt = PromptTemplate(input_variables=["question", "answer"], template="Question: {question}\n{answer}")

    template = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        suffix="Question: {input}",
        input_variables=["input"]
    )

    prompt = template.format(input=human_input)

    llm = OpenAI()

    llm.invoke(prompt)    

    output = llm.invoke(prompt)    

    return output

#Langchain>Modules>Model I/O>Prompt>PromptTemplate>FewShotChatModel
#https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/few_shot_examples
def _Modules_FewShotChatModelPromptTemplate(human_input):    
    
    from langchain.prompts import (
        ChatPromptTemplate,
        FewShotChatMessagePromptTemplate,
    )
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import LLMChain

    chat_model = ChatOpenAI()
    
    examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
    ]
    
    example_prompt = ChatPromptTemplate.from_messages([
        ("human","{input}"),
        ("ai","{output}"),
    ]
    )

    template = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", "You are a wondrous wizard of math."),
        template,
        ("human", "{input}"),
      ]
    )

    '''
    llm_chain  = LLMChain(
            llm=chat_model,
            prompt=final_prompt,
            verbose=True,
        )

    output = llm_chain.predict ()
    '''

    chain = final_prompt | chat_model

    output = chain.invoke({"input": human_input})

    return output


#Langchain>Modules>Model I/O>Prompt>PromptTemplate>Dynamic FewShot Prompting
#https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/few_shot_examples
def _Modules_DynamicFewShotPrompt(human_input):    
    
    from langchain.chat_models import ChatOpenAI  
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma

    from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    SemanticSimilarityExampleSelector
    )

    examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
    {"input": "2+4", "output": "6"},
    {"input": "What did the cow say to the moon?", "output": "nothing at all"},
    {
        "input": "Write me a poem about the moon",
        "output": "One for the moon, and one for me, who are we to talk about the moon?",
    },
    ]

    to_vectorize = [" ".join(example.values()) for example in examples]
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)

    example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,
    )

    # The prompt template will load examples by passing the input do the `select_examples` method
    example_selector.select_examples({"input": "horse"})

    # Define the few-shot prompt.
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        # The input variables select the values to pass to the example_selector
        input_variables=["input"],
        example_selector=example_selector,
        # Define how each example will be formatted.
        # In this case, each example will become 2 messages:
        # 1 human, and 1 AI
        example_prompt=ChatPromptTemplate.from_messages(
            [("human", "{input}"), ("ai", "{output}")]
        ),
    )

    final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a wondrous wizard of math."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
    )

    chat_model = ChatOpenAI()

    chain = final_prompt | chat_model

    output = chain.invoke({"input": human_input})

    return output