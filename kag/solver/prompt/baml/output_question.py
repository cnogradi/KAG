from typing import List


from kag.interface import PromptABC
from .baml_client.sync_client import b

@PromptABC.register("default_output_question")
class OutputQuestionPrompt(PromptABC):

    template_zh = "oq"

    template_en = "oq"

    @property
    def template_variables(self) -> List[str]:
        return ["context", "question"]

    def parse_response(self, response: str, **kwargs):
        response = b.OutputQuestion(kwargs['context'], kwargs['question'])
        logger.debug("multihopgen:{}".format(response))
        return response.answer
