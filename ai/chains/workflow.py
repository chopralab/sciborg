'''
Custom LLM chains for workflow construction
'''
from langchain_core.prompts import PromptTemplate

from pydantic import BaseModel as BaseModelV2
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.language_models import BaseLanguageModel

from langchain_openai import ChatOpenAI

from typing import Type

from sciborg.core.library.base import BaseCommandLibrary

from sciborg.ai.schema.workflow import RunWorkflowSchemaV1
from sciborg.ai.prompts.workflow import (
    BASE_WORKFLOW_CONSTRUCTION_PROMPT,
    BASE_WORKFLOW_PLANNING_PROMPT,
)

def create_workflow_planner_chain(
    library: BaseCommandLibrary,
    llm: BaseLanguageModel = ChatOpenAI(temperature=0.1),
    prompt: str = BASE_WORKFLOW_PLANNING_PROMPT
) -> RunnableSequence:
    '''
    Creates an LLM chain for workflow planning based on the provided command library.

    This chain can serve as an intermediary between higher level planning and operational planning.
    
    Returns a RunnableSequence (LCEL chain) instead of LLMChain for LangChain v1.0+ compatibility.
    '''
    # Define prompt template
    prompt_template = PromptTemplate(
        template=prompt,
        input_variables=['query'],
        partial_variables={
            'command_library': library.model_dump_json(indent=2)
        }
    )

    # Return LCEL chain (prompt | llm)
    return prompt_template | llm

def create_workflow_constructor_chain(
    library: BaseCommandLibrary,
    llm: BaseLanguageModel = ChatOpenAI(temperature=0.1),
    prompt: str = BASE_WORKFLOW_CONSTRUCTION_PROMPT,
    output_schema: Type[BaseModelV2] = RunWorkflowSchemaV1,
) -> RunnableSequence:
    '''
    Builds a LLM chain for operational workflow construction based on a the provided command library.
    
    Returns a RunnableSequence (LCEL chain) instead of LLMChain for LangChain v1.0+ compatibility.
    '''
    # Define output parser
    output_parser = JsonOutputParser(pydantic_object=output_schema)

    # Define prompt template
    prompt_template = PromptTemplate(
        template=prompt,
        input_variables=['query'],
        partial_variables={
            "command_library": library.model_dump_json(indent=2),
            "format_instructions": output_parser.get_format_instructions(),
            }
    )

    # Return LCEL chain (prompt | llm | parser)
    return prompt_template | llm | output_parser