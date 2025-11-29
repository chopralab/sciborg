'''
Uses Langchains built-in JSON parsing to build SCIBORG objects in a highly structure manner
'''
from typing import Any, Dict, Type, List

from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel as BaseModelV2, Field
from langchain_classic.memory import ConversationBufferWindowMemory

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable, RunnableConfig, chain, RunnableSequence

from langchain_classic.base_memory import BaseMemory

from langchain_openai import ChatOpenAI

from sciborg.ai.schema.parameter import ParameterSchemaV1
from sciborg.ai.schema.command import LibraryCommandSchemaV1
from sciborg.ai.schema.workflow import RunWorkflowSchemaV1

class LinqxLLMChain(LLMChain):
    '''
    Custom LLM chain for SCIBORG object utility.

    This object will operate as a standard Langchain LLM chain but uses Langchain's built-in JSON parser to
    format output in compliance with SCIBORG object format.

    To build an object invoke the LLM chain with a specific query

    ```python
    # Construct the LLM chain
    chain = LinqxLLMChain(
        llm=llm,
        sciborg_object=LinqxObjectClass
    )

    # Run the chain against a query
    output = chain.invoke(input={'query': 'Some information about the object you would like to build...'})

    # Get the output and build the object from the mapping
    sciborg_object_dictionary = output['text']
    sciborg_object = LinqxObjectClass(**sciborg_object_dictionary)
    ```

    If the LLM is not able to get a correct version of the object after the first iteration, you can recall the chain with
    the output and error message to attempt to fix the issue

    ```python
    try:
        sciborg_object_dictionary = output['text'] 
        sciborg_object = LinqxObjectClass(**sciborg_object_dictionary)
    except Exception as e:
        new_output = chain.invoke(
            input = {
                'query': 'Some information about the object you would like to build...'},
                'past_response': json.dumps(sciborg_object_dictionary),
                'error' = str(e)
            }
        )
    ```
    '''
    sciborg_object: Type[BaseModelV2]
    prompt: PromptTemplate = PromptTemplate(
        template='Answer the users query.\n{query}',
        input_variables=['query']
    )

    def model_post_init(self) -> None:
        '''
        Initialize output parser and update prompt with format instructions.
        Uses Pydantic v2 model_post_init hook.
        '''
        # Assign output parser
        self.output_parser = JsonOutputParser(pydantic_object=self.sciborg_object)

        # Create new set of partial varaibles
        new_partial_varaibles = self.prompt.partial_variables
        new_partial_varaibles['format_instructions'] = self.output_parser.get_format_instructions()

        # Create new prompt
        self.prompt = PromptTemplate(
            template=self.prompt.template+"\n{format_instructions}",
            input_variables=self.prompt.input_variables,
            partial_variables=new_partial_varaibles
        )

    def invoke(
        self,
        input: Dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        # We are reinvoking with the past response, restructure the prompt for this
        if 'past_response' in input.keys() and 'past_response' not in self.prompt.input_variables:
            new_input_vars = [var for var in self.prompt.input_variables]
            new_input_vars.append('past_response')
            self.prompt = PromptTemplate(
                template=self.prompt.template+"""\nThe previous output you generated to the question was the following:\n{past_response}.""",
                input_variables=new_input_vars,
                partial_variables=self.prompt.partial_variables
            )
        # If we are reinvoking with an error, restructure the prompt for this
        if 'error' in input.keys() and 'error' not in self.prompt.input_variables:
            new_input_vars = [var for var in self.prompt.input_variables]
            new_input_vars.append('error')
            self.prompt = PromptTemplate(
                template=self.prompt.template+"""\nThe previous output generated an error. 
                please fix this error in your response.\nError:\n{error}""",
                input_variables=new_input_vars,
                partial_variables=self.prompt.partial_variables
            )
        
        # Return LLM chain invoke
        return super().invoke(input, config, **kwargs)

def create_json_parser(
    pydantic_object: BaseModelV2, 
    llm: BaseLanguageModel | None = None
) -> RunnableSequence:
    '''
    Creates an LLM based JSON parser for a pydantic schema using Langchains internal
    JSON output parser and LLM chain.

    Parameters
    ```
    pydantic_object: BaseModel # Must be a base model from pydantic version < 2
    llm: BaseLanguageModel | None = None # Defaults to GPT-4 if None
    ```
    
    Returns
    ```
    return RunnableSequence # A pipeline that parses user queries into a JSON format conforming to the schema
    ```
    '''
    if llm is None:
        llm = ChatOpenAI(model='gpt-4')
    
    parser = JsonOutputParser(pydantic_object=pydantic_object)
    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | llm | parser

def create_sciborg_parameter_parser(llm: BaseLanguageModel | None = None) -> RunnableSequence:
    '''
    Creates a parser for SCIBORG parameters.
    
    Returns a RunnableSequence (LCEL chain) for LangChain v1.0+ compatibility.
    '''
    if llm is not None:
        return create_json_parser(ParameterSchemaV1, llm)
    else:
        return create_json_parser(ParameterSchemaV1)
    
def create_sciborg_command_parser(llm: BaseLanguageModel | None = None) -> RunnableSequence:
    '''
    Creates a parser for SCIBORG commands.
    
    Returns a RunnableSequence (LCEL chain) for LangChain v1.0+ compatibility.
    '''
    if llm is not None:
        return create_json_parser(LibraryCommandSchemaV1, llm)
    else:
        return create_json_parser(LibraryCommandSchemaV1)
    
def create_sciborg_workflow_parser(llm: BaseLanguageModel | None = None) -> RunnableSequence:
    '''
    Creates a parser for SCIBORG workflows.
    
    Returns a RunnableSequence (LCEL chain) for LangChain v1.0+ compatibility.
    '''
    if llm is not None:
        return create_json_parser(RunWorkflowSchemaV1, llm)
    else:
        return create_json_parser(RunWorkflowSchemaV1)