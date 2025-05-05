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

import json
from string import Template
from typing import List
from kag.common.conf import KAG_PROJECT_CONF
from kag.interface import PromptABC
from knext.reasoner.client import ReasonerClient

from .baml_client.sync_client import b

@PromptABC.register("baml_question_ner")
class QuestionNER(PromptABC):

    """English template string"""
    template_en: str = "qner"
    """Chinese template string"""
    template_zh: str = "qner"

    def __init__(self, language: str = "", **kwargs):
        super().__init__(language, **kwargs)
        self.schema = (
            ReasonerClient(project_id=KAG_PROJECT_CONF.project_id)
            .get_reason_schema()
            .keys()
        )
        self.template = Template(self.template).safe_substitute(schema=self.schema)

    @property
    def template_variables(self) -> List[str]:
        return ["input"]

    def parse_response(self, response: str, **kwargs):

        response = b.QuestionNER(list(self.schema), kwargs['input']).dict()
        rsp = response
        if isinstance(rsp, str):
            rsp = json.loads(rsp)
        if isinstance(rsp, dict) and "output" in rsp:
            rsp = rsp["output"]
        if isinstance(rsp, dict) and "named_entities" in rsp:
            entities = rsp["named_entities"]
        else:
            entities = rsp

        return entities
