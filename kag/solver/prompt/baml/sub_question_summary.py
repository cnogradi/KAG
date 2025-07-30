from typing import List

from kag.interface import PromptABC

from .baml_client.sync_client import b

@PromptABC.register("baml_sub_question_summary")
class SubQuestionSummary(PromptABC):

    template_zh = "zh"

    template_en = "en"

    @property
    def template_variables(self) -> List[str]:
        return ["history", "question", "knowledge_graph", "docs"]

    def parse_response(self, response: str, **kwargs):
        response = b.SubQuestionSummary(kwargs['history'], kwargs['question'], kwargs['knowledge_graph'], kwargs['docs'])
        return response.answer
