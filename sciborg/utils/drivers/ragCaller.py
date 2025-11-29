"""
This is the module for retrieving external information and generating an answer using a RAG (Retrieval-Augmented Generation) approach.
It uses a vector store to retrieve relevant documents based on the user's question and then generates an answer using a language model.
# The code is designed to be modular and reusable, allowing for easy integration with different vector stores and language models.
"""



import bs4
from langchain_classic import hub
from langchain_core.documents import Document
from typing import List, Any
from typing_extensions import Annotated, TypedDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel

import numpy as np

# LangGraph is optional - this module requires it
try:
    from langgraph.graph import START, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    START = None
    StateGraph = None


template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""


# Desired schema for response
class AnswerWithSources(TypedDict):
    """An answer to the question, with sources."""
    answer: str
    sources: Annotated[
        List[str],
        ...,
        "List of sources (author + year) used to answer the question",
    ]


def _load_vector_store(path: str, embeddings: Any):
    from langchain_community.vectorstores import FAISS
    # If the embeddings argument is the string "openai", instantiate the corresponding embeddings.
    if embeddings == 'openai':
        from langchain_openai import OpenAIEmbeddings
        embedding_instance = OpenAIEmbeddings()
    else:
        embedding_instance = embeddings
    return FAISS.load_local(path, embedding_instance, allow_dangerous_deserialization=True).as_retriever()


class State(TypedDict):
    question: str
    context: List[Document]
    answer: AnswerWithSources


def _create_rag_agent_executor(
    vectorstore_path: str, 
    embeddings: Any, 
    llm: BaseLanguageModel | None = None
) -> StateGraph:
    """
    Create and return an agent executor using a RAG approach.
    
    NOTE: This function requires langgraph. Install with: pip install langgraph
    
    Parameters:
      - vectorstore_path: Path to load the vector store from.
      - llm: A language model object supporting an .invoke() method.
      - embeddings: Embedding model or the string 'openai' to use for vector store retrieval.
    
    Returns:
      - A compiled StateGraph that first retrieves context documents based on the question,
        then generates an answer using the retrieved context.
    
    Raises:
      ImportError: If langgraph is not installed
    """
    if llm is None:
        llm = ChatOpenAI(temperature=0.1)
    
    if not LANGGRAPH_AVAILABLE:
        raise ImportError(
            "LangGraph is required for this function. Install with: pip install langgraph"
        )
    # Load the vector store from disk, passing the embeddings argument.
    vector_store = _load_vector_store(vectorstore_path, embeddings)
    
    # Load a prompt for RAG from the hub.
    prompt = PromptTemplate.from_template(template)
    
    # Define the retrieval step: retrieve relevant documents using similarity search.
    def retrieve(state: State) -> dict:
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}
    
    # Define the generation step: combine the retrieved documents and question into a prompt for the LLM.
    def generate(state: State) -> dict:
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        # Depending on your LangChain version, you might use .format() instead of .invoke()
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        structured_llm = llm.with_structured_output(AnswerWithSources)
        response = structured_llm.invoke(messages)
        return {"answer": response}
    
    # Build the state graph with the two sequential steps.
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    
    return graph


def external_information_retrieval(
    question: str, 
    vectorstore_path: str, 
    embeddings: Any,
    llm: BaseLanguageModel | None = None
) -> dict:
    """
    Retrieve external information and generate an answer using a RAG approach.
    
    Parameters:
      - question: The question to be answered.
      - vectorstore_path: Path to load the vector store from. Usually provided by the user.
      - llm: A language model object supporting an .invoke() method.
      - embeddings: Embedding model or the string 'openai' to use for vector store retrieval.
    
    Returns:
      - A structured answer with sources.
    """

    # Creating the executor.
    executor = _create_rag_agent_executor(vectorstore_path, embeddings, llm)
    result = executor.invoke({"question": question})
    return {"answer": result['answer']}