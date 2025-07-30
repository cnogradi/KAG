from string import Template
from typing import List

from kag.interface import PromptABC

from .baml_client.sync_client import b

@PromptABC.register("baml_solve_question")
class SolveQuestion(PromptABC):

    template_zh = "sq"

    template_en = "sq"

    @property
    def template_variables(self) -> List[str]:
        return ["history", "question", "knowledge_graph", "docs"]

    def parse_response(self, response: str, **kwargs):
        response = b.SolveQuestion(kwargs['question'], kwargs['history'], kwargs['docs'], kwargs['knowledge_graph'])
        return response.answer
