from typing import List


from kag.interface import PromptABC

from .baml_client.sync_client import b

@PromptABC.register("baml_summary_question")
class SummaryQuestionWithOutSPO(PromptABC):

    template_zh = "zh"

    template_en = "en"

    @property
    def template_variables(self) -> List[str]:
        return ["history", "question", "docs"]

    def parse_response(self, response: str, **kwargs):
        response = b.QuestionSummary(kwargs['history'], kwargs['question'], kwargs['docs'])
        return response.answer
