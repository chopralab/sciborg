from langchain_classic.memory.summary import ConversationSummaryMemory, SummarizerMixin
from langchain_classic.memory.chat_memory import BaseChatMemory
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_classic.chains import LLMChain
from typing import Dict, Any, List, Type
from sciborg.ai.prompts.memory import EMBEDDING_SUMMARY_PROMPT

class EmbeddingSummaryMemory(ConversationSummaryMemory):
    '''
    TODO implement this to summarize information access
    '''
    filtered_tool_list : List[str]
    prompt: BasePromptTemplate = EMBEDDING_SUMMARY_PROMPT
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        #filtering
        filtered_tools = [action for action in outputs['intermediate_steps'] if action[0].tool in self.filtered_tool_list]
        copied_outputs = outputs.copy()
        copied_outputs['intermediate_steps'] = str(filtered_tools)
        return super().save_context(inputs, copied_outputs)