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
from concurrent.futures import ThreadPoolExecutor, as_completed
from kag.interface import (
    ReaderABC,
    ExtractorABC,
    SplitterABC,
    VectorizerABC,
    SinkWriterABC,
    KAGBuilderChain,
)

from kag.examples.MusiquePike.builder.decomposer_abc import DecomposerABC
from kag.common.utils import generate_hash_id

logger = logging.getLogger(__name__)

@KAGBuilderChain.register("kag_atomic_chunk_extractor_chain")
class KagAtomicChunkExtractorChain(KAGBuilderChain):
    def __init__(
            self,
            reader: ReaderABC,
            splitter: SplitterABC,
            decomposer: DecomposerABC = None,
            vectorizer: VectorizerABC = None,
            writer: SinkWriterABC = None
    ):
        """
        Initializes the DefaultUnstructuredBuilderChain instance.

        Args:
            reader (ReaderABC): The reader component to be used.
            extractor (ExtractorABC): The extractor component to be used.
            vectorizer (VectorizerABC): The vectorizer component to be used.
            writer (SinkWriterABC): The writer component to be used.
        """
        self.reader = reader
        self.splitter = splitter
        self.decomposer = decomposer
        self.vectorizer = vectorizer
        self.writer = writer

    def build(self, **kwargs):
        pass

    def invoke(self, input_data, max_workers=10, **kwargs):
        """
        Invokes the builder chain to process the input file.

        Args:
            input_data: The path to the input file to be processed.
            max_workers (int, optional): The maximum number of threads to use. Defaults to 10.
            **kwargs: Additional keyword arguments.

        Returns:
            List: The final output from the builder chain.
        """

        def execute_node(node, node_input, **kwargs):
            if not isinstance(node_input, list):
                node_input = [node_input]
            node_output = []
            for item in node_input:
                node_output.extend(node.invoke(item, **kwargs))
            return node_output

        def run_extract(chunk):
            flow_data = [chunk]
            input_key = chunk.hash_key

            for node in [
                self.decomposer,
                self.vectorizer,
                self.writer,
            ]:
                if node is None:
                    continue
                flow_data = execute_node(node, flow_data, key=input_key)

            if len(flow_data) > 0:
                return {input_key: flow_data[0]}

        reader_output = self.reader.invoke(input_data, key=generate_hash_id(input_data))
        splitter_output = []

        for chunk in reader_output:
            splitter_output.extend(self.splitter.invoke(chunk, key=chunk.hash_key))

        processed_chunk_keys = kwargs.get("processed_chunk_keys", set())
        filtered_chunks = []
        processed = 0
        for chunk in splitter_output:
            if chunk.hash_key not in processed_chunk_keys:
                filtered_chunks.append(chunk)
            else:
                processed += 1
        logger.debug(
            f"Total chunks: {len(reader_output)}. Checkpointed: {processed}, Pending: {len(filtered_chunks)}."
        )

        result = []
        with ThreadPoolExecutor(max_workers) as executor:
            futures = [executor.submit(run_extract, chunk) for chunk in filtered_chunks]

            from tqdm import tqdm

            for inner_future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Decomposition Atomic Query From Chunk",
                    position=1,
                    leave=False,
            ):
                ret = inner_future.result()
                result.append(ret)
        return result