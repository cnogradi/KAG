import re
from string import Template
from typing import List
import logging

from kag.interface import PromptABC

from .baml_client.sync_client import b

logger = logging.getLogger(__name__)


@PromptABC.register("baml_resp_extractor")
class RespExtractor(PromptABC):
    template_zh = "re"
    template_en = "re"
    @property
    def template_variables(self) -> List[str]:
        return ["supporting_fact", "instruction"]

    def parse_response(self, response: str, **kwargs):
        response = b.RespExtractor(kwargs['supporting_fact'], kwargs['instruction'])
        logger.debug("推理器判别:{}".format(response))
        return response.passage
