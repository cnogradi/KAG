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

@PromptABC.register("baml_thought_then_answer")
class ThoughtThenAnswerPrompt(PromptABC):

    template_en = "tt"
    template_zh = "tt"

    @property
    def template_variables(self) -> List[str]:
        return ["docs", "cur_question", "questions"]

    def parse_response(self, response: str, **kwargs):
        response = b.ToughtThenAnswer(kwargs['cur_question'], kwargs['questions'], kwargs['docs'] )        
        return f"""
        Thought:
        
        {response.thought}

        Answer:

        {response.answer}
        """
