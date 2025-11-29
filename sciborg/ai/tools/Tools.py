# Import langchain classes/functions
from typing import (
    Any,
    Literal,
    List,
    Dict,
    Type
)
# from langchain.agents.agent_toolkits import JsonToolkit
from langchain_core.tools import StructuredTool, Tool, tool
from langchain_community.tools import HumanInputRun
from langchain_community.tools import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool
)
from langchain_community.tools.file_management.write import WriteFileInput
from langchain_core.callbacks.manager import CallbackManagerForToolRun
# from langchain.pydantic_v1 import BaseModel, PositiveInt, Field
from pydantic import BaseModel as BaseModelV2, Field, PositiveInt
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools import BaseTool

# Spython import for Apptainer interaction
from spython.main import Client

# Builtin Python imports
import sys, os
import json

# Import SCIBORG infrastructure objects
from sciborg.core.parameter.base import ParameterModel, ValueType
from sciborg.core.command.base import BaseDriverCommand

# Import internal V1 schema
from sciborg.ai.schema.parameter import ParameterSchemaV1, ParameterModelSchemaV1
from sciborg.ai.schema.scheduler import ScheduleSchemaV1

# Note: BaseScheduleTemplate and Apptainer schemas are not currently defined in the codebase
# These features may need to be implemented or removed from usage
# BaseScheduleTemplate (referenced in scheduler.py but doesn't exist)
# ApptainerFilenameAppTemplateV1
# ApptainerFilenameListTemplateV1
# ApptainerFilenameTemplateV1 


######################################################
# Environment Variable Validation (Lazy)
######################################################

def _get_env_var(var_name: str) -> str:
    """Get environment variable, raising error if not set."""
    value = os.environ.get(var_name)
    if value is None:
        raise EnvironmentError(f"Please set environment variable '{var_name}'")
    if not os.path.isdir(value):
        raise NotADirectoryError(f"Directory {value} does not exist, please check '{var_name}'")
    return value

def _get_llm_json_dir() -> str:
    """Get LLM_JSON_DIR with lazy validation."""
    return _get_env_var('LLM_JSON_DIR')

def _get_llm_microservice_dir() -> str:
    """Get LLM_MICROSERVICE_DIR with lazy validation."""
    return _get_env_var('LLM_MICROSERVICE_DIR')

#######################################################
# Custom Python Functions
#######################################################

# Custom functions for SCIBORG interaction
# Note: Using StructuredTool.from_function instead of @tool decorator
# to avoid import-time issues with environment variables
def build_parameter_model_func(**kwargs) -> str:
    return ParameterModel(**kwargs).model_dump_json(indent=2, exclude_defaults=True)

def build_schedule_template_func(**kwargs) -> str:
    # TODO: BaseScheduleTemplate doesn't exist - needs to be implemented or removed
    # For now, use ScheduleSchemaV1
    return ScheduleSchemaV1(**kwargs).model_dump_json(indent=2, exclude_defaults=True)

def build_request_from_template_func(**kwargs) -> str:
    # TODO: BaseScheduleTemplate doesn't exist - needs to be implemented or removed
    # For now, use ScheduleSchemaV1
    return ScheduleSchemaV1(**kwargs).model_dump_json(indent=2)

# Custom functions for Spython interaction
def get_apptainer_microservice_tags(filename: str) -> str:
    filename = _get_llm_microservice_dir() + '/' + filename
    return Client.inspect(filename)['attributes']['labels']['TAGS'].strip()

def get_apptainer_microservice_help(filename: str, app: str | None = None) -> str:
    filename = _get_llm_microservice_dir() + '/' + filename
    if app is None:
        return Client.inspect(filename)['attributes']['helpfile'].strip()
    else:
        return Client.inspect(filename, app=app)['attributes']['apps'][app]['helpfile'].strip()
    
def get_apptainer_microservice_app_input(filename: str, app: str) -> str:
    filename = _get_llm_microservice_dir() + '/' + filename
    return Client.inspect(filename, app=app)['attributes']['apps'][app]['labels']['INPUT']

def get_apptainer_microservice_app_output(filename: str, app: str) -> str:
    filename = _get_llm_microservice_dir() + '/' + filename
    return Client.inspect(filename, app=app)['attributes']['apps'][app]['labels']['OUTPUT']

def get_all_apps(filename: str) -> str:
    filename = _get_llm_microservice_dir() + '/' + filename
    return Client.inspect(filename)['attributes']['labels']['APPS']

def get_all_app_help(filename: str) -> str:
    apps = Client.inspect(_get_llm_microservice_dir() + '/' + filename)['attributes']['labels']['APPS'].split(", ")
    all_app_help = {}
    for app in apps:
        all_app_help[app] = {
            'helpfile': get_apptainer_microservice_help(filename, app),
            'input': get_apptainer_microservice_app_input(filename, app),
            'output': get_apptainer_microservice_app_output(filename, app)
        }
    return json.dumps(all_app_help, indent=2)

def get_all_app_help_from_files(filenames: List[str]) -> str:
    all_file_apps = {}
    for filename in filenames:
        all_file_apps[filename] = json.loads(get_all_app_help(filename))
    return json.dumps(all_file_apps, indent=2)

def get_tags_from_filenames(filenames: List[str]) -> str:
    all_tags = {}
    for filename in filenames:
        all_tags[filename] = get_apptainer_microservice_tags(filename)
    return json.dumps(all_tags, indent=2)

def get_help_from_filenames(filenames: List[str]) -> str:
    all_help = {}
    for filename in filenames:
        all_help[filename] = get_apptainer_microservice_help(filename)
    return json.dumps(all_help, indent=2)

def get_all_tags(**kwargs) -> str:
    sif_file_list = [file for file in os.listdir(_get_llm_microservice_dir()) if file.endswith('.sif')]
    return get_tags_from_filenames(sif_file_list)

def get_all_help(**kwargs) -> str:
    sif_file_list = [file for file in os.listdir(_get_llm_microservice_dir()) if file.endswith('.sif')]
    return get_help_from_filenames(sif_file_list)

##########################################################
# Tool from Python Functions
##########################################################

# Define custom SCIBORG infrastructure tools
# Note: build_parameter_model_func uses ParameterModel, not ParameterModelSchemaV1
# The schema should match what ParameterModel expects
parameter_init_tool = StructuredTool.from_function(
    args_schema=ParameterModelSchemaV1,
    func=build_parameter_model_func,
    name='Build Parameter',
    description='''
        Builds a parameter model.
        The data types of upper_limit, lower_limit, and allowed_values must match the parameter data type.
        For example, if the data type is a float, they must be passed in as floats not integers. 
        Precision is only required if data_type is a float. It should be an integer greater than 0.
        If upper and lower limits are provided, upper limit must be greater than or equal to the lower limit.
    '''
)

schedule_template_init_tool = StructuredTool.from_function(
    args_schema=ScheduleSchemaV1,
    func=build_schedule_template_func,
    name='Build Schedule Template',
    description='''
        Builds a schedule template.
        The argument schema for this tool is recursively defined. If a python dictionary is passed in as 
        a value to the template dictionary, it must be in the same format as the provided argument schema.
    '''
)

build_request_from_template_tool = StructuredTool.from_function(
    args_schema=ScheduleSchemaV1,
    func=build_request_from_template_func,
    name='Build Request from Template',
    description='''
        Converts a valid schedule template to a request.
        The input to this tool should be a valid schedule template output by the 'Build Schedule Template' tool.
        You should always provide the user with the output of this tool and tell them that this is the template which
        they created.
    '''
)

# Define custom tools for interacting with containerized microservices
get_all_microservice_tag_tool = StructuredTool.from_function(
    func=get_all_tags,
    name='Get_All_Microservice_Tags',
    description='''
        Gets tags for all available microservices.
        The output from this tool will be a JSON formatted string which contains "filename": "list of tags"
    '''
)

get_all_microservice_help_tool = StructuredTool.from_function(
    func=get_all_help,
    name='Get_All_Microservice_Help',
    description='''
        Gets the helpfile information for all available microservices.
        The output from this tool will be a JSON formatted string which contains "filename": "helpfile information"
    '''
)

get_microservice_all_app_help_tool = StructuredTool.from_function(
    # args_schema=ApptainerFilenameAppTemplateV1,  # TODO: Schema doesn't exist
    func=get_all_app_help,
    name='Get_Microservice_All_App_Help',
    description='''
        Gets the help information for all applications of the provided microservice file.
        The input to this tool should be a dictionary of a single .sif filename for which you want to get information on the apps.
        The output from this tool will be a JSON formatted string which contains "app name": "help information"
    '''
)

get_microservice_all_app_help_from_files_tools = StructuredTool.from_function(
    # args_schema=ApptainerFilenameListTemplateV1,  # TODO: Schema doesn't exist
    func=get_all_app_help_from_files,
    name='Get_Microservice_All_App_Help_From_Files',
    description='''
        Gets the help information for all applications of the provided microservice .sif files.
        The input to this tool should be a dictionary of a a list of .sif filenames you want to get information on the apps.
        The output from this tool will be a JSON formatted string which contains "filename": "app name": "help information", "input information", "output information"
    '''
)

get_microservice_help_from_filenames_tool = StructuredTool.from_function(
    # args_schema=ApptainerFilenameListTemplateV1,  # TODO: Schema doesn't exist
    func=get_help_from_filenames,
    name='Get_Microservice_Help_From_File_List',
    description='''
        Gets the helpfile information for the provided files.
        The input to this tool should be a dictionary of a list of .sif filenames you want to get the tags of.
        The output from this tool will be a JSON formatted string which contains "filename": "list of tags"
    '''
)

get_microservice_tags_from_filenames_tool = StructuredTool.from_function(
    # args_schema=ApptainerFilenameListTemplateV1,  # TODO: Schema doesn't exist
    func=get_tags_from_filenames,
    name='Get_Microservice_Tags_From_File_List',
    description='''
        Gets the tag information for the provided files.
        The input to this tool should be a list of .sif filenames you want to get the tags of.
        You must provide a filename that was observed from the output of the 'Get_All_Microservice_Tags' tool.
        The output from this tool will be a JSON formatted string which contains "filename": "list of tags"
    '''
)

###############################################################
# Custom Tools
###############################################################

class JSONWriteFileInput(WriteFileInput):

    text: str | Dict = Field(..., description='text to write to file')

class JSONWriteFileTool(WriteFileTool):
    args_schema: Type[BaseModelV2] = JSONWriteFileInput
    description: str = "Write JSON formatted file to disk"

    def _run(
            self, 
            file_path: str,
            text: str | Dict,
            append: bool = False,
            run_manager: CallbackManagerForToolRun | None = None
    ) -> str:
        if isinstance(text, dict):
            text = json.dumps(text, indent=2)
        return super()._run(file_path, text, append, run_manager)

# Define human tool
human_tool = HumanInputRun()

# Define custom read/write file tools (lazy initialization)
# These will be initialized when first accessed, not at import time
_write_file_tool = None
_read_file_tool = None
_list_dir_tool = None
_list_microservice_dir_tool = None

def _get_write_file_tool():
    global _write_file_tool
    if _write_file_tool is None:
        _write_file_tool = JSONWriteFileTool(root_dir=_get_llm_json_dir())
    return _write_file_tool

def _get_read_file_tool():
    global _read_file_tool
    if _read_file_tool is None:
        _read_file_tool = ReadFileTool(root_dir=_get_llm_json_dir())
    return _read_file_tool

def _get_list_dir_tool():
    global _list_dir_tool
    if _list_dir_tool is None:
        _list_dir_tool = ListDirectoryTool(root_dir=_get_llm_json_dir())
    return _list_dir_tool

def _get_list_microservice_dir_tool():
    global _list_microservice_dir_tool
    if _list_microservice_dir_tool is None:
        _list_microservice_dir_tool = ListDirectoryTool(root_dir=_get_llm_microservice_dir())
    return _list_microservice_dir_tool

# For backward compatibility, these will be initialized lazily when toolkits are accessed
# Access them via the getter functions or through toolkits
write_file_tool = None  # Will be initialized on first toolkit access
read_file_tool = None
list_dir_tool = None
list_microservice_dir_tool = None

###############################################################
# Toolkits
###############################################################

# Define toolkits

# Template Builder Toolkit
'''
 - Tool for parameter construction
 - Tool for schedule template construction
 - Tool for asking the human for guidance
 - Tool for writing a file to disk
'''
def _get_template_builder_tools():
    """Get template builder tools, initializing file tools lazily."""
    global write_file_tool
    if write_file_tool is None:
        write_file_tool = _get_write_file_tool()
    return [
        parameter_init_tool,
        schedule_template_init_tool,
        human_tool,
        write_file_tool
    ]

def get_template_builder_tools():
    """Get template builder tools."""
    return _get_template_builder_tools()

def get_template_builder_tool_names():
    """Get template builder tool names."""
    return [tool.name for tool in _get_template_builder_tools()]

# For backward compatibility, try to create as lists (will fail if env vars not set)
try:
    template_builder_tools = _get_template_builder_tools()
    template_builder_tool_names = [tool.name for tool in template_builder_tools]
except (EnvironmentError, NotADirectoryError):
    # If env vars not set, these will be None and should be accessed via getter functions
    template_builder_tools = None
    template_builder_tool_names = None

# Request Builder Toolkit
'''
 - Tool for asking the human for guidance
 - Tool for listing a provided directory
 - Tool for reading in a file from disk
 - Tool for building a request from a provided template
'''
def _get_request_builder_tools():
    """Get request builder tools, initializing file tools lazily."""
    global list_dir_tool, read_file_tool
    if list_dir_tool is None:
        list_dir_tool = _get_list_dir_tool()
    if read_file_tool is None:
        read_file_tool = _get_read_file_tool()
    return [
        human_tool,
        list_dir_tool,
        read_file_tool,
        build_request_from_template_tool
    ]

def get_request_builder_tools():
    """Get request builder tools."""
    return _get_request_builder_tools()

def get_request_builder_tool_names():
    """Get request builder tool names."""
    return [tool.name for tool in _get_request_builder_tools()]

# For backward compatibility, try to create as lists (will fail if env vars not set)
try:
    request_builder_tools = _get_request_builder_tools()
    request_builder_tool_names = [tool.name for tool in request_builder_tools]
except (EnvironmentError, NotADirectoryError):
    # If env vars not set, these will be None and should be accessed via getter functions
    request_builder_tools = None
    request_builder_tool_names = None

# Microservice Selector Toolkit
'''
 - Tool for getting tags for all microservices in the microservice directory
 - Tool for getting help information of all microservices
'''
microservice_selector_tools = [
    get_all_microservice_tag_tool,
    get_microservice_help_from_filenames_tool
]
microservice_selector_tool_names = [tool.name for tool in microservice_selector_tools]

#App Selector Toolkit
'''
 - Tool for getting tags for all microservices in the microservice directory.
 - Tool for getting help information of all microservices.
 - Tool for getting information on the app endpoints of a given microservice.
'''
app_selector_tools = [
    get_all_microservice_tag_tool,
    get_microservice_help_from_filenames_tool,
    get_microservice_all_app_help_from_files_tools
]
app_selector_tool_names = [tool.name for tool in app_selector_tools]