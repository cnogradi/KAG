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
from typing import List

from kag.interface import PromptABC

from .baml_client.sync_client import b

from .baml_client.types import NamedEntities

@PromptABC.register("baml_std")
class OpenIEEntitystandardizationdPrompt(PromptABC):


    """English template string"""
    template_en: str = "std"
    """Chinese template string"""
    template_zh: str = "std"

    @property
    def template_variables(self) -> List[str]:
        return ["input", "named_entities"]

    def parse_response(self, response: str, **kwargs):

        response = b.ExtractNamedEntities(kwargs['input'], json.dumps(kwargs['named_entities'])).dict()

        rsp = response
        if isinstance(rsp, str):
            rsp = json.loads(rsp)
        if isinstance(rsp, dict) and "output" in rsp:
            rsp = rsp["output"]
        if isinstance(rsp, dict) and "entities" in rsp:
            standardized_entity = rsp["entities"]
        else:
            standardized_entity = rsp

        entities_with_offical_name = set()
        merged = []
        entities = kwargs.get("named_entities", [])
        for entity in standardized_entity:
            merged.append(entity)
            entities_with_offical_name.add(entity["name"])
        # in case llm ignores some entities
        for entity in entities:
            if entity["name"] not in entities_with_offical_name:
                entity["official_name"] = entity["name"]
                merged.append(entity)
        return merged
