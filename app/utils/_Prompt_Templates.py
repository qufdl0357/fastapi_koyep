def _Modules_PromptTemplate(adjective, content):    

    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate

    llm = OpenAI()

    prompt_template = PromptTemplate.from_template("Tell me a {adjective} joke about {content}.")
    prompt = prompt_template.format(adjective=adjective, content=content)

    result = llm(prompt)

    return result