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
# flake8: noqa
import json
from typing import List
from kag.interface import PromptABC

from .baml_client.sync_client import b

@PromptABC.register("default_query_rewrite")
class QueryRewritePrompt(PromptABC):
    template_zh = "qwp"

    template_en = "qwp"

    @property
    def template_variables(self) -> List[str]:
        return ["context", "query"]

    def parse_response(self, response: str, **kwargs):
        response = b.OutputQuestion(kwargs['context'], kwargs['question'])
        return response.query
