from typing import List
import logging

from kag.interface import PromptABC

from kag.examples.kjv.solver.baml_client.sync_client import b

logger = logging.getLogger(__name__)


@PromptABC.register("baml_summary_resp_generator")
class RespGenerator(PromptABC):
    #
    # This prompt template is adapted from LightRAG:
    #
    #   https://github.com/HKUDS/LightRAG/blob/45cea6e/lightrag/prompt.py#L156
    #
    # which can produce answers with better comprehensiveness, diversity
    # and empowerment to general questions.
    #
    # NOTE: This prompt template may not be the best for all the tasks.
    #       For example, it won't produce answers with high EM and F1
    #       scores for the hotpotqa, 2wiki and musique datasets.
    #
    _prompt_template = "as"

    template_zh = _prompt_template.format(language="Chinese")
    template_en = _prompt_template.format(language="English")

    @property
    def template_variables(self) -> List[str]:
        return ["memory", "instruction"]

    def parse_response(self, response: str, **kwargs):
        response = b.AnswerQuestion(kwargs['memory'], kwargs['instruction'])
        return response.answer
