import re
from string import Template
from typing import List
import logging

from kag.interface import PromptABC

from .baml_client.sync_client import b

logger = logging.getLogger(__name__)


@PromptABC.register("baml_resp_verifier")
class RespVerifier(PromptABC):
    template_zh = "rv"
    template_en = "rv"

    @property
    def template_variables(self) -> List[str]:
        return ["sub_instruction", "supporting_fact"]

    def parse_response(self, response: str, **kwargs):
        response = b.RespVerifier(kwargs['sub_instruction'], kwargs['supporting_fact'])
        logger.debug("推理器判别:{}".format(response))
        if response.answered:
            return f"The answer to the Question'{kwargs['sub_instruction'],}' is '{response.answer}'"
        return None
