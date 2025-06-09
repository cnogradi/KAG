import logging
from typing import List

from kag.interface import PromptABC

from .baml_client.sync_client import b

logger = logging.getLogger(__name__)


@PromptABC.register("baml_rewrite_sub_task_query")
class DefaultRewriteSubTaskQueryPrompt(PromptABC):
    template_zh = "drtq"
    template_en = "drtq"

    def is_json_format(self):
        return False

    @property
    def template_variables(self) -> List[str]:
        return ["content", "input"]

    def parse_response(self, response: list, **kwargs):
        response = b.RewriteSubTaskQuery(kwargs['content'],kwargs['input'])
        logger.debug(f"rewrite sub query:{response}")
        return response
