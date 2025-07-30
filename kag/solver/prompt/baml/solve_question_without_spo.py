from string import Template
from typing import List


from kag.interface import PromptABC

from .baml_client.sync_client import b

@PromptABC.register("baml_solve_question_without_spo")
class SolveQuestionWithOutSPO(PromptABC):

    template_zh = "sqs"

    template_en = "sqs"

    @property
    def template_variables(self) -> List[str]:
        return ["history", "question", "docs"]

    def parse_response(self, response: str, **kwargs):
        response = b.SolveQuestionNoSPO(kwargs['question'], kwargs['history'], kwargs['docs'])
        return response.answer
