import logging
import re
from typing import List

from kag.interface import PromptABC

from .baml_client.sync_client import b
logger = logging.getLogger(__name__)


@PromptABC.register("baml_logic_form_plan")
class LogicFormPlanPrompt(PromptABC):

    template_zh = "lfp"
    template_en = "lfp"

    @property
    def template_variables(self) -> List[str]:
        return ["question"]

    def parse_response(self, response: str, **kwargs):
        try:
            response = b.LogicFromPlan(kwargs['question'])
            logger.debug(f"logic form:{response}")
            return [ plan.step for plan in response.plans], [ plan.action for plan in response.plans]
        except Exception as e:
            logger.warning(f"{response} parse logic form faied {e}", exc_info=True)
            return [], []
