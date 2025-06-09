from kag.interface import PromptABC
import logging
from typing import List

from .baml_client.sync_client import b

logger = logging.getLogger(__name__)


@PromptABC.register("baml_deduce_extractor")
class DeduceExtractor(PromptABC):
    template_zh = "de"
    template_en = "de"
    @property
    def template_variables(self) -> List[str]:
        return ["memory", "instruction"]

    def parse_response(self, response: str, **kwargs):
        response = b.DeduceExtractor(kwargs['instruction'], kwargs['memory'])
        logger.debug("Reasoner discrimination:{}".format(response))
        return response.no_relevant_information, response.extractor
