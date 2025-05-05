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
import re
from typing import List

from kag.interface import PromptABC

from .baml_client.sync_client import b
from baml_py.errors import BamlClientHttpError
import logging
import spacy
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_md")

@PromptABC.register("baml_semantic_seg")
class BAMLSemanticSegPrompt(PromptABC):
    template_zh = "ssp"
    template_en = "ssp"

    @property
    def template_variables(self) -> List[str]:
        return ["input"]

    def find_similar_fragment(self, text, fragment, similarity_threshold=0.7):
        doc = nlp(text)
        fragments = [sent.text for sent in doc.sents]
        embeddings = [nlp(fragment).vector for fragment in fragments]
        similarity = 0
        similar_fragment = ('', 0)
        for i in range(len(fragments)):
            frag_sim = cosine_similarity([embeddings[i]], [nlp(fragment).vector])[0][0]
            if frag_sim > similarity_threshold and frag_sim > similarity:
                similar_fragment = (fragments[i], similarity)
        return similar_fragment

    def get_segments(self, content, seg_info, last_loc):

        for attempt in range(3):
            try:
                response = b.SemanticSegment(content)      
                break
            except Exception as e:
                if attempt:
                    logger.info(f"Caught exception: {e}, retrying on {attempt} atempt")
                    sleep(10)
                    continue
                raise e

        segments = response.segments

        for index, segment in enumerate(response.segments):
            starting_point = segment.starting_point.strip()
            next_loc = -1
            if index + 1 < len(response.segments):
                next_loc = content.find(response.segments[index+1].starting_point.strip(), last_loc)
            summary = segment.summary.strip()

            if not starting_point:
                logger.info(f"Segment has no starting point for summary {summary}")
                continue  # Skip segments with missing starting or ending points

            # Find the *first* occurrence of the starting point *after* the last known location
            loc = content.find(starting_point, last_loc)

            # If still not found, try lowercase matching
            if loc == -1:
                logger.info(f"Trying lowercase for: {segment}")
                loc = content.lower().find(starting_point.lower(), last_loc)

            if loc == -1:
                # Trying with clarifier
                logger.info(f"Trying clarification/similarity using last_loc {last_loc} and next_loc {next_loc}")
                c = content[last_loc:next_loc] if next_loc != -1 else content[last_loc:]
                s = segment.starting_point.strip()
                logger.info(f"Trying similiarity search for: {segment} with content {c} and segment {s}")
                fragment, similarity = self.find_similar_fragment(c, s)
                if similarity:
                    logger.info(f"Found similarity: {fragment} for {s} with similarity {similarity}")
                    loc = content.find(fragment, last_loc)
                if loc == -1:
                    logger.info(f"Trying clarification for: {segment} with content {c} and segment {s}")
                    clarify_response = b.SemanticSegmentClarify(c, s)
                    loc = content.find(clarify_response.starting_point.strip(), last_loc)
                    if loc == -1:
                        logger.info(f"Searching from begining in case LLMs gave us answers out of order for {segment}")
                        loc = content.find(starting_point)
                        if loc == -1:
                            logger.info(f"incorrect seg: {segment} classified as {clarify_response}")
                            # Try to recurse through the remainder hoping for a better draw
                            self.get_segments(content, seg_info, last_loc)
                            break
            logger.info(f"found location: {loc} with starting pt len {len(starting_point)}")
            seg_info.append((loc, summary))
            last_loc = loc + len(starting_point)

    def parse_response(self, response: str, **kwargs):
        """
        Parses the semantic segmentation response to extract locations and summaries.

        Args:
            response (str): The original input string.
            kwargs (dict): A dictionary containing the input string and the output list (segmentation results).

        Returns:
            list: A list of tuples, where each tuple contains the start location and summary for each segment.
                Returns an empty list if the input is invalid or the segmentation results are empty.
        """
        content = kwargs.get("input", "")

        if not content:
            return []

        seg_info = []

        last_loc = 0

        self.get_segments(content, seg_info, last_loc)

        return seg_info


