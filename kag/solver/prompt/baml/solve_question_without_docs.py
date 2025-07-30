from string import Template
from typing import List

from kag.interface import PromptABC

from .baml_client.sync_client import b

@PromptABC.register("baml_solve_question_without_docs")
class SolveQuestionWithOutDocs(PromptABC):

    template_zh = "sqd"

    template_en = "sqd"

    @property
    def template_variables(self) -> List[str]:
        return ["history", "question", "knowledge_graph"]

    def parse_response(self, response: str, **kwargs):
        response = b.SolveQuestionNoDocs(kwargs['question'], kwargs['history'], kwargs['knowledge_graph'])
        return response.answer
