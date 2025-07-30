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
import json
from typing import List
from kag.interface import PromptABC, Task

from .baml_client.sync_client import b

logger = logging.getLogger()

@PromptABC.register("baml_thought_iterative_planning")
class DefaultIterativePlanningPrompt(PromptABC):
    example_executors = [
        {
            "name": "Retriever",
            "description": "Retrieve relevant knowledge from the local knowledge base.",
            "parameters": {
                "query": {
                    "type": "string",
                    "description": "User-provided query for retrieval.",
                    "optional": False,
                },
            },
        },
        {
            "name": "Math",
            "description": "Used to address users' math or computational problems.",
            "parameters": {
                "query": {
                    "type": "string",
                    "description": "The computable problem derived from the user's input question, retaining the essential information for the calculation target and dependencies.",
                    "optional": False,
                }
            },
        },
        {
            "name": "Deduce",
            "description": "Synthesizes precise, evidence-backed answers to user queries by analyzing provided contextual documents. Note: Contextual documents are pre-loaded and processed implicitly; no explicit context parameter is required.",
            "parameters": {
                "query": {
                    "type": "string",
                    "description": "User-provided query.",
                    "optional": False,
                },
            },
        },
        {
            "name": "Finish",
            "description": "Performs no operation and is solely used to indicate that the task has been completed.",
            "parameters": {},
        },
    ]
    template_zh = 'dip'

    template_en = "dip"

    @property
    def template_variables(self) -> List[str]:
        return ["context", "executors", "query"]

    def parse_response(self, response: str, **kwargs):
        response = b.ToughtIterativePlanning(example_executors, kwargs['query'], kwargs['context'], kwargs['executors'])
        task = Task(
            executor=response.name,
            arguments={ argument.name:argument.instruction for argument in response.arguments },
            thought=response.thought,
        )
        logging.info(f'{executor["arguments"]} thought {executor.get("thought", "")}')
        return [task]