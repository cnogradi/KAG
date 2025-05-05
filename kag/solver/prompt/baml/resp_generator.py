import re
from string import Template
from typing import List
import logging

from kag.interface import PromptABC

from .baml_client.sync_client import b

logger = logging.getLogger(__name__)


@PromptABC.register("baml_resp_generator")
class RespGenerator(PromptABC):
    template_zh = "rg"
    template_en = "rg"

    @property
    def template_variables(self) -> List[str]:
        return ["memory", "instruction"]

    def parse_response(self, response: str, **kwargs):
        response = b.RespGenerator(kwargs['memory'], kwargs['instruction']).dict()
        logger.debug("推理器判别:{}".format(response))
        return response
