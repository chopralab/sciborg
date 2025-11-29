"""
RAG (Retrieval Augmented Generation) Agent

A specialized agent for answering questions using a vector database of documents.
This agent retrieves relevant information from embeddings and provides answers with citations.
"""

from langchain_classic.agents.format_scratchpad import format_to_openai_function_messages
from langchain_classic.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_classic.agents import tool, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseLanguageModel


def rag_agent(
    question: str,
    path_to_embeddings: str,
    llm: BaseLanguageModel | None = None
) -> AgentExecutor:
    """
    Create a RAG (Retrieval Augmented Generation) agent.
    
    This agent can answer questions by retrieving relevant information from a vector
    database of documents. It provides answers with citations (title and page numbers).
    
    Args:
        question: The question to answer (used for initialization)
        path_to_embeddings: Path to the FAISS vector database directory
        llm: Language model to use (defaults to GPT-4 with temperature=0.1)
        
    Returns:
        AgentExecutor: Configured agent ready to answer questions using RAG
        
    Note:
        Currently assumes FAISS vector store. The embeddings are loaded using
        OpenAIEmbeddings. To use a different vector store, modify this function.
        
    Example:
        ```python
        agent = rag_agent("What is the procedure?", "/path/to/embeddings")
        result = agent.invoke({"question": "What is the procedure?"})
        ```
    """
    if llm is None:
        llm = ChatOpenAI(temperature=0.1)
    
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(
        path_to_embeddings, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever()

    @tool
    def get_answer_from_information(situation: str) -> str:
        """
        Retrieve relevant information from the document database.
        
        The user will provide a situation/question and the tool will return
        relevant answers from the information with citations.
        
        Args:
            situation: The question or situation to search for
            
        Returns:
            List of answers with citations (title and page)
        """
        results = retriever.invoke(situation)
        answers = []
        for result in results:
            answer = {
                'citation_title': result.metadata.get('source', 'Unknown'),
                'citation_page': result.metadata.get('page', 'Unknown'),
                'answer': result.page_content
            }
            answers.append(answer)
        return answers

    tools = [get_answer_from_information]

    llm_with_tools = llm.bind(functions=[convert_to_openai_function(t) for t in tools])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a powerful assistant with access to a document database.
                
                You can answer questions about user queries using relevant sources. 
                You must provide factual answers based ONLY on the retrieved information.
                
                Process:
                    1. First use the tool to retrieve relevant information
                    2. Analyze the retrieved information carefully
                    3. Provide a clear answer based ONLY on the retrieved information
                    4. Return the answer with respective citations (title and page) 
                       present in the retrieved information
                """,
            ),
            ("user", "{question}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    chain = (
        {
            "question": lambda x: x["question"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
    )

    rag_agent_executor = AgentExecutor(agent=chain, tools=tools, verbose=True)

    return rag_agent_executor
