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
from kag.interface import PromptABC, Task

from .baml_client.sync_client import b

@PromptABC.register("baml_retriever_static_planning")
class RetrieverStaticPlanningPrompt(PromptABC):
    template_zh = "sp"
    template_en = "sp"

    @property
    def template_variables(self) -> List[str]:
        return ["executors", "query"]

    def parse_response(self, response: str, **kwargs):
        response = b.RetrieverStaticPlanning(kwargs['executors'], kwargs['query'])
        logger.debug(f"retriever static planning prompt:{response}")
        return Task.create_tasks_from_dag({task.task_id:{
            "executor": task.executor,
            "dependent_task_ids": task.dependent_task_ids,
            "arguments": {argument.name: argument.instruction for argument in task.arguments},
        } for task in response.tasks})
