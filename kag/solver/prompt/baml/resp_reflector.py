import logging
from typing import List

from kag.interface import PromptABC

from .baml_client.sync_client import b

logger = logging.getLogger(__name__)


@PromptABC.register("baml_resp_reflector")
class RespReflector(PromptABC):
    template_zh = "rr"
    template_en = "rr"

    @property
    def template_variables(self) -> List[str]:
        return ["memory", "instruction"]

    def parse_response(self, response: str, **kwargs):
        response = b.RespReflector(kwargs['memory'], kwargs['instruction'])
        logger.debug("infer result:{}".format(response))
        return response.thought_questions
