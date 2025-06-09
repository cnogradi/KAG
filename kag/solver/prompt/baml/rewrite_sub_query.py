import logging
from typing import List

from kag.interface import PromptABC

from .baml_client.sync_client import b

logger = logging.getLogger(__name__)


@PromptABC.register("default_rewrite_sub_query")
class DefaultRewriteSubQuery(PromptABC):
    template_zh = "rwsq"
    template_en = "rwsq"

    @property
    def template_variables(self) -> List[str]:
        return ["history_qa", "question"]

    def parse_response(self, response: str, **kwargs):
        response = b.RewriteSubQuery(kwargs['history_qa'],kwargs['question'])
        logger.debug(f"rewrite sub query:{response}")
        return response
