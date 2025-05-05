import logging
from typing import List

from kag.interface import PromptABC

from .baml_client.sync_client import b

from .baml_client.types import Choice

logger = logging.getLogger(__name__)

@PromptABC.register("baml_deduce_judge")
class DeduceJudge(PromptABC):

    """English template string"""
    template_en: str = "dj"
    """Chinese template string"""
    template_zh: str = "dj"

    @property
    def template_variables(self) -> List[str]:
        return ["memory", "instruction"]

    def parse_response(self, response: str, **kwargs):
        response = b.DeduceJudge(kwargs['instruction'], kwargs['memory'])
        logger.debug("Reasoner judgment:{}".format(response))
        return response.no_available_information, response.assessable
