from langchain.pydantic_v1 import BaseModel, root_validator
from typing import List, Any, Dict

from ASPIRE_LINQX.core.workflow.base import BaseInfoWorkflow
from ASPIRE_LINQX.ai.schema.command import BaseRunCommandSchemaV1

class RunWorkflowSchemaV1(BaseModel):
    name: str
    commands: List[BaseRunCommandSchemaV1]