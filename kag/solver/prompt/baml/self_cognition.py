from typing import List

from kag.common.conf import KAG_PROJECT_CONF
from kag.interface import PromptABC

from .baml_client.sync_client import b

@PromptABC.register("baml_self_cognition")
class SelfCognitionPrompt(PromptABC):

    template_zh = "sc"

    template_en = "sg"

    @property
    def template_variables(self) -> List[str]:
        return ["question"]

    def parse_response(self, response: str, **kwargs):
        try:
            response = b.SelfCognition(kwargs['question'])
            logger.debug("Self Cognition:{}".format(response))
            return response.is_cognition_question
        except Exception as e:
            print(e)
            return False
