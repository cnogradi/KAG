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
from kag.interface import PromptABC

@PromptABC.register("judger_prompt")
class JudgerPrompt(PromptABC):

    template_en = """
        # Task
        Providing a question and its correct answer labels, your task is to analyze whether a given answer is correct or not. An answer is treated as correct if the meaning of any label is expressed in the answer. Redundant expression in answer is allowed.
        
        # Question
        $question
        
        # Correct Answer Labels
        $gold
        
        # Answer that Require Judgment
        $prediction
        
        Is the answer correct or not? You output should only be "Yes" or "No".
        """.strip()

    template_zh = template_en

    def __init__(self, language: str = "", **kwargs):
        super().__init__(language, **kwargs)

    @property
    def template_variables(self) -> List[str]:
        return ["question", "gold", "prediction"]

    def parse_response(self, response: str, **kwargs):
        return response.strip().lower()
