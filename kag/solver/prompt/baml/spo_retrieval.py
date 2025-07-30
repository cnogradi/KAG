import logging
from typing import List, Dict

from kag.interface import PromptABC

from .baml_client.sync_client import b

logger = logging.getLogger(__name__)


@PromptABC.register("baml_spo_retrieval")
class SpoRetrieval(PromptABC):
    template_zh = "zh"
    template_en = "en"

    @property
    def template_variables(self) -> List[str]:
        return ["question", "mention", "candis"]

    def parse_response(self, response, **kwargs):
        response = b.SPORetrieval(kwargs['question'], kwargs['mention'], kwargs['candis'])
        logger.debug(response)
        return response.spos
