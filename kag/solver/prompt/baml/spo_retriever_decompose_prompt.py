# -*- coding: utf-8 -*-
# Copyright 2023 OpenSPG Authors
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied.
import logging
import re
from typing import List
from kag.interface import PromptABC

from .baml_client.sync_client import b

logger = logging.getLogger()


@PromptABC.register("baml_spo_retriever_decompose")
class DefaultSPORetrieverDecomposePrompt(PromptABC):
    instruct_zh = "dsprd"
    default_case_zh = "dc"

    instruct_en = "dsprd"

    default_case_en = "dc"

    def __init__(self, **kwargs):
        self.template_zh = f"zh"
        self.template_en = f"en"
        super().__init__(**kwargs)

    @property
    def template_variables(self) -> List[str]:
        return ["question"]

    def parse_response(self, response: str, **kwargs):
        try:
            response = b.SPORetrievalDecompose(kwargs['question'])
            logger.debug(f"spo retrieval decompose form:{response}")
            return [ spo.step for spo in response.spos], [ spo.action for spo in response.spos]
        except Exception as e:
            logger.warning(f"{response} parse logic form failed {e}", exc_info=True)
            return [], []