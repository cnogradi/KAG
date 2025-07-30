from typing import List
import logging

from kag.common.utils import get_now
from kag.interface import PromptABC

from .baml_client.sync_client import b

logger = logging.getLogger(__name__)


@PromptABC.register("baml_without_refer_generator_prompt")
class WithOutReferGeneratorPrompt(PromptABC):
    template_zh = 'wrg'
    template_en = "wrg"

    @property
    def template_variables(self) -> List[str]:
        return ["content", "query", "ref"]

    def parse_response(self, response: str, **kwargs):
        response = b.WithoutRefGenerator(get_now(language='en'), kwargs['content'], kwargs['query'])
        logger.debug("Reasoner discrimination:{}".format(response))
        return response.answer
