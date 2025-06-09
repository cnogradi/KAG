import logging
from typing import List

from kag.common.utils import get_now
from kag.interface import PromptABC

from .baml_client.sync_client import b

logger = logging.getLogger(__name__)


@PromptABC.register("baml_expression_builder")
class ExpressionBuildr(PromptABC):
    template_zh = "eb"
    template_en = "eb"

    @property
    def template_variables(self) -> List[str]:
        return ["question", "context", "error"]

    def parse_response(self, response: str, **kwargs):
        response = b.ExpressionBuilder(get_now(language='en'), kwargs['question'], kwargs['context'], kwargs['error'])
        logger.debug("Expression builder:{}".format(response))
        if not response.not_known:
            return response.python_code
        return ""
