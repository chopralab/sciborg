from typing import Dict, Any, Coroutine, Callable, Tuple, Type
from inspect import signature, getdoc
from pydantic import ValidationError as ValidationErrorV2 # Our infrastructure uses Pydantic V2, which will throw this ValidationError
import re

from langchain_core.tools import BaseTool
# from langchain.pydantic_v1 import root_validator
from pydantic import model_validator
# from langchain.pydantic_v1 import ValidationError as ValidationErrorV1
from langchain_core.tools import create_schema_from_function, ToolException

from sciborg.core.command.base import BaseDriverCommand

class LinqxTool(BaseTool):
    '''

    '''
    name: str = "undefined"
    description: str = "undefined"
    sciborg_command:  BaseDriverCommand
    handle_tool_error: bool | str | Callable[[ToolException], str] | None = True
    errors_caught: Tuple[Type[Exception]] = (TypeError, ValueError, KeyError, ValidationErrorV2)

    @model_validator(mode='before')
    def validate_tool(cls, values: Dict[str, Any]):
        '''
        Dynamically sets name, description, and args_schema based on the provided SCIBORG command.
        
        Uses Pydantic v2 model_validator for LangChain v1.0+ compatibility.
        '''
        values['name'] = values['sciborg_command'].name
        values['description'] = f"Function Signature:\n{signature(values['sciborg_command']._function)}\nFunction Docstring:\n{getdoc(values['sciborg_command']._function)}"
        values['args_schema'] = create_schema_from_function(
            model_name=f"{values['sciborg_command'].name}_",
            func=values['sciborg_command']._function
        )
        return values
    
    @staticmethod
    def _sanatize_error(error_message: str) -> str:
        '''
        Private method for sanatizing error message by removing extraneous information
        that is not relevent to LLM operation.

        Currently removes
        - Any URL matching regex: `r'https?://\S+|www\.\S+'`
        - The phrase used in Pydantic ValidationError's `'For further information visit'`
        '''
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        sanatized_e = url_pattern.sub('', str(error_message))
        sanatized_e = sanatized_e.replace('For further information visit', '')
        sanatized_e = sanatized_e.rstrip()
        return sanatized_e

    def _run(self, **kwargs) -> str:
        '''
        The tool calls the SCIBORG command with keyword arguments provided by the LLM in a try/catch statement.

        Default errors caught are:
        
        ```python
        (TypeError, ValueError, KeyError, ValidationError)
        ```

        After catching an error, a ToolException is raised with the message of the caught error.
        '''
        try: 
            return str(self.sciborg_command(**kwargs))
        except (TypeError, ValueError, KeyError, ValidationErrorV2) as e:
            raise ToolException(LinqxTool._sanatize_error(str(e)))
        
    def _arun(self, *args: Any, **kwargs: Any) -> Coroutine[Any, Any, Any]:
        '''
        TODO: implement async tool running when this feature is needed
        '''
        raise NotImplementedError("Not currently supported for SCIBORG tools")