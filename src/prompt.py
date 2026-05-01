from langchain_core.prompts import ChatPromptTemplate

def get_prompt():
    return ChatPromptTemplate.from_template(
        "You are a medical assistant.\n\n"
        "Answer ONLY using the given context.\n"
        "If the answer is NOT in the context, say:\n"
        "'I don't know based on the given information.'\n\n"
        "Do NOT guess.\n"
        "Do NOT give general advice.\n"
        "Do NOT add extra information.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{input}\n\n"
        "Answer:"
    )