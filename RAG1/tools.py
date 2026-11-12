from .indexing import vectorstore

from langchain.tools import tool

last_retrieved_context = []

@tool
def collection_info(query : str) -> str:
    """
    Search the PDF knowledge base and return relevant information
    for the user's question
    """
    global last_retrieved_context

    docs = vectorstore.similarity_search(query, k = 2)
    last_retrieved_context = [doc.page_content for doc in docs]

    return '\n\n'.join(last_retrieved_context)

#print(collection_info("black and white pattern"))