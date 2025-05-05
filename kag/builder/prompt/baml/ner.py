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
from knext.schema.client import SchemaClient

from .baml_client.sync_client import b

from .baml_client.types import Entities


@PromptABC.register("baml_ner")
class OpenIENERPrompt(PromptABC):

    """English template string"""
    template_en: str = "ner"
    """Chinese template string"""
    template_zh: str = "ner"

    def __init__(self, language: str = "", **kwargs):
        super().__init__(language, **kwargs)
        self.schema = SchemaClient(
            project_id=KAG_PROJECT_CONF.project_id
        ).extract_types()
        self.template = Template(self.template).safe_substitute(
            schema=json.dumps(self.schema)
        )

    @property
    def template_variables(self) -> List[str]:
        return ["input"]

    def parse_response(self, response: str, **kwargs):

        response = b.ExtractEntities(kwargs['input'], json.dumps(self.schema)).dict()
        
        rsp = response
        if isinstance(rsp, str):
            rsp = json.loads(rsp)
        if isinstance(rsp, dict) and "output" in rsp:
            rsp = rsp["output"]
        if isinstance(rsp, dict) and "entities" in rsp:
            entities = rsp["entities"]
        else:
            entities = rsp

        return entities
