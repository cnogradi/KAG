from typing import List
import logging

from kag.common.utils import get_now
from kag.interface import PromptABC

from .baml_client.sync_client import b

logger = logging.getLogger(__name__)


@PromptABC.register("baml_refer_generator_prompt")
class ReferGeneratorPrompt(PromptABC):
    template_zh = "rg"
    template_en = "rg"

    @property
    def template_variables(self) -> List[str]:
        return ["content", "query", "ref"]

    def parse_response(self, response: str, **kwargs):
        response = b.ReferenceGenerator(get_now(language='en'), kwargs['content'], kwargs['ref'], kwargs['query'])
        logger.debug("refernce_generator:{}".format(response))
        return f'{response.answer}\n<reference id="{response.reference_id if not response.internal_knowledge else "internal knowledge"}"></reference>\n'
