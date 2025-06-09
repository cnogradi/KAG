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
from typing import List
from kag.interface import PromptABC

from .baml_client.sync_client import b

@PromptABC.register("default_multi_hop_generator")
class MultiHopGeneratorPrompt(PromptABC):

    template_en = "mhg"
    template_zh = template_en

    @property
    def template_variables(self) -> List[str]:
        return ["content", "query"]

    def is_json_format(self):
        return True

    def parse_response(self, response: dict, **kwargs):
        response = b.MultiHopGenerator(kwargs['content'], kwargs['query'])
        logger.debug("multihopgen:{}".format(response))
        return response.answer