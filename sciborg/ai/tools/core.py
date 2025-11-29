from typing import Dict, Any, Coroutine, Callable, Tuple, Type, get_type_hints
from inspect import signature, getdoc, Parameter
from pydantic import ValidationError as ValidationErrorV2, create_model, Field
import re

from langchain_core.tools import BaseTool
from pydantic import model_validator
from langchain_core.tools import ToolException

from sciborg.core.command.base import BaseDriverCommand


def create_args_schema_from_function(func: Callable, model_name: str) -> Type:
    """
    Create a Pydantic model from a function signature.
    This is a custom implementation that avoids issues with langchain's create_schema_from_function.
    """
    sig = signature(func)
    hints = {}
    try:
        hints = get_type_hints(func)
    except Exception:
        pass
    
    fields = {}
    for param_name, param in sig.parameters.items():
        if param_name in ('self', 'cls'):
            continue
        
        # Get the type annotation
        param_type = hints.get(param_name, str)
        
        # Handle default values
        if param.default is Parameter.empty:
            fields[param_name] = (param_type, ...)
        else:
            fields[param_name] = (param_type, param.default)
    
    return create_model(model_name, **fields)


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
        # Use custom schema creation to avoid langchain compatibility issues
        values['args_schema'] = create_args_schema_from_function(
            func=values['sciborg_command']._function,
            model_name=f"{values['sciborg_command'].name}_"
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

    @staticmethod
    def _is_schema_dict(value: Any) -> bool:
        '''
        Check if a value looks like a JSON Schema definition rather than an actual value.
        Schema dicts typically have 'type' and optionally 'title' keys.
        '''
        if not isinstance(value, dict):
            return False
        schema_keys = {'type', 'title', 'description', 'properties', 'required'}
        return bool(schema_keys.intersection(value.keys())) and 'type' in value
    
    @staticmethod
    def _clean_schema_from_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        '''
        Remove any kwargs that look like JSON Schema definitions.
        This handles edge cases where schema metadata is passed instead of values.
        '''
        cleaned = {}
        for key, value in kwargs.items():
            if LinqxTool._is_schema_dict(value):
                # Skip schema-like dictionaries - they shouldn't be passed as values
                continue
            cleaned[key] = value
        return cleaned

    def _run(self, tool_input: str | dict[str, Any] | None = None, **kwargs) -> str:
        '''
        The tool calls the SCIBORG command with keyword arguments provided by the LLM in a try/catch statement.

        Default errors caught are:
        
        ```python
        (TypeError, ValueError, KeyError, ValidationError)
        ```

        After catching an error, a ToolException is raised with the message of the caught error.
        '''
        # Handle both dict input and keyword arguments
        if tool_input is not None:
            if isinstance(tool_input, dict):
                # Clean out any schema-like values before merging
                cleaned_input = self._clean_schema_from_kwargs(tool_input)
                kwargs.update(cleaned_input)
            elif isinstance(tool_input, str):
                # If string input, try to parse or use as single argument
                # This shouldn't happen with structured tools, but handle gracefully
                pass
        
        # Also clean kwargs directly in case schema values were passed there
        kwargs = self._clean_schema_from_kwargs(kwargs)
        
        try: 
            return str(self.sciborg_command(**kwargs))
        except (TypeError, ValueError, KeyError, ValidationErrorV2) as e:
            raise ToolException(LinqxTool._sanatize_error(str(e)))
        
    def _arun(self, *args: Any, **kwargs: Any) -> Coroutine[Any, Any, Any]:
        '''
        TODO: implement async tool running when this feature is needed
        '''
        raise NotImplementedError("Not currently supported for SCIBORG tools")
