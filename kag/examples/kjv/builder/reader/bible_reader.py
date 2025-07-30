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

import os
import re
from typing import List

from kag.builder.model.chunk import Chunk
from kag.interface import ReaderABC
from kag.common.utils import generate_hash_id
from knext.common.base.runnable import Input, Output


@ReaderABC.register("bible")
@ReaderABC.register("bible_reader")
class BibleReader(ReaderABC):
    """
    A class for parsing text files or text content into Chunk objects.

    This class inherits from ReaderABC and provides the functionality to read text content,
    whether it is from a file or directly provided as a string, and convert it into a list of Chunk objects.
    """

    def _invoke(self, input: Input, **kwargs) -> List[Output]:
        """
        The main method for processing text reading. This method reads the content of the input (which can be a file path or text content) and converts it into chunks.

        Args:
            input (Input): The input string, which can be the path to a text file or direct text content.
            **kwargs: Additional keyword arguments, currently unused but kept for potential future expansion.

        Returns:
            List[Output]: A list containing Chunk objects, each representing a piece of text read.

        Raises:
            ValueError: If the input is empty.
            IOError: If there is an issue reading the file specified by the input.
        """
        if not input:
            raise ValueError("Input cannot be empty")

        try:
            if os.path.exists(input):
                chunks = []
                chapter = ''
                with open(input, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line or line == '\n':
                            continue
                        match = re.search(r'Book ([0-9]*) (.*)', line)
                        if match:
                            if chapter:
                                print(f'Processing new {book} chapter {previous_chapter} as {chapter}')
                                chunk = Chunk(
                                    id=generate_hash_id(chapter),
                                    name=f'{book} {previous_chapter}',
                                    content=chapter,
                                )
                                chunks.append(chunk)
                            # New book
                            book_num = match.group(1)
                            book = match.group(2)
                            chapter = ""
                            previous_chapter = '001'
                        else:
                            match = re.search(r'([0-9]{2}):([0-9]{3}):([0-9]{3}) (.*)', line)
                            if match:
                                # New verse
                                book_num_from_verse = match.group(1)
                                chapter_num = match.group(2)
                                verse_num = match.group(3)
                                if previous_chapter != chapter_num:
                                    print(f'Processing {book} chapter {previous_chapter} as {chapter}')
                                    # New chapter
                                    chunk = Chunk(
                                        id=generate_hash_id(chapter),
                                        name=f'{book} {previous_chapter}',
                                        content=chapter,
                                    )
                                    chunks.append(chunk)
                                    previous_chapter = chapter_num
                                    chapter = match.group(4)
                                else:
                                    chapter = chapter + ' ' + match.group(4)
                            else:
                                chapter = chapter + ' ' + line[:-1]
                    # Add last chapter
                    chunk = Chunk(
                        id=generate_hash_id(chapter),
                        name=f'{book} {previous_chapter}',
                        content=chapter,
                    )
                    chunks.append(chunk)
                    return chunks
            else:
                return ValueError("Input must be a file")
        except OSError as e:
            raise IOError(f"Failed to read file: {input}") from e
