# from langchain.pydantic_v1 import BaseModel, root_validator, Field
from pydantic import BaseModel as BaseModelV2, Field, field_validator
from typing import Dict, Type, Any, Union
import json
from uuid import UUID

from sciborg.ai.schema.parameter import(
    ParameterSchemaV1,
    ParameterModelSchemaV1,
)
from sciborg.core.command.base import BaseInfoCommand

class LibraryCommandSchemaV1(BaseModelV2):

    # Command Model attributes
    name: str
    microservice: str
    desc: str = Field('A short description of the command')
    uuid: UUID
    parameters: Dict[str, ParameterModelSchemaV1] | None = Field(default={})
    
    @field_validator('parameters', mode='before')
    @classmethod
    def normalize_parameters(cls, v):
        """Normalize None to empty dict for parameters."""
        if v is None:
            return {}
        return v
    # Library command attributes
    has_return: bool = False
    return_signature: Dict[str, str] | None = Field(
        default={},
        description="""
        A return signature in JSON format of the command
        Key - name of the return varaible
        Value - description of the return varaible
        """
    )
    
class BaseRunCommandSchemaV1(BaseModelV2):
    name: str
    microservice: str
    uuid: UUID
    # desc: str = ""
    parameters: Dict[str, ParameterSchemaV1] | None = Field(default={})
    
    @field_validator('parameters', mode='before')
    @classmethod
    def normalize_parameters(cls, v):
        """Normalize None to empty dict for parameters."""
        if v is None:
            return {}
        return v
    
    save_vars: Dict[str, str] = {}