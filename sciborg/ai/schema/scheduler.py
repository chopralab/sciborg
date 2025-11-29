# from langchain.pydantic_v1 import BaseModel, root_validator
from pydantic import BaseModel as BaseModelV2, model_validator
from typing import Dict, Any

from sciborg.ai.schema.parameter import ParameterSchemaV1

class ScheduleSchemaV1(BaseModelV2):
    name: str
    template:  Dict[str, Dict | ParameterSchemaV1]

    @model_validator(mode='before')
    def validate_scheduler(cls, values: Dict[str, Any]) -> Dict:
        # TODO: BaseScheduleTemplate doesn't exist in sciborg.core.scheduler.base
        # Validation removed until BaseScheduleTemplate is implemented
        # BaseScheduleTemplate(**values)
        return values