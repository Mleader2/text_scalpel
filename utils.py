# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Utility functions for LaserTagger."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import json
from bert import tokenization
import tensorflow as tf

### 中文用word piece,　保留空格
class my_tokenizer_class(object):
    def __init__(self, vocab_file, do_lower_case):
        self.full_tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=do_lower_case)

    # 需要包装一下，因为如果直接对中文用full_tokenizer.tokenize，会忽略文本中的空格
    def tokenize(self, text):
        segments = text.split(" ")
        word_pieces = []
        for segId, segment in enumerate(segments):
            if segId > 0:
                word_pieces.append(" ")
            word_pieces.extend(self.full_tokenizer.tokenize(segment))
        return word_pieces

    def convert_tokens_to_ids(self, tokens):
        id_list = [self.full_tokenizer.vocab[t]
                   if t != " " else self.full_tokenizer.vocab["[unused20]"] for t in tokens]
        return id_list


def yield_sources_and_targets(
        input_file,
        input_format):
    """Reads and yields source lists and targets from the input file.

    Args:
      input_file: Path to the input file.
      input_format: Format of the input file.

    Yields:
      Tuple with (list of source texts, target text).
    """
    if input_format == 'wikisplit':
        yield_example_fn = _yield_wikisplit_examples
    elif input_format == 'discofuse':
        yield_example_fn = _yield_discofuse_examples
    else:
        raise ValueError('Unsupported input_format: {}'.format(input_format))

    for sources, target in yield_example_fn(input_file):
        yield sources, target


def _yield_wikisplit_examples(
        input_file):
    # The Wikisplit format expects a TSV file with the source on the first and the
    # target on the second column.
    with tf.gfile.GFile(input_file) as f:
        for line in f:
            source, target, lcs_rate = line.rstrip('\n').split('\t')
            yield [source], target


def _yield_discofuse_examples(
        input_file):
    """Yields DiscoFuse examples.

    The documentation for this format:
    https://github.com/google-research-datasets/discofuse#data-format

    Args:
      input_file: Path to the input file.
    """
    with tf.gfile.GFile(input_file) as f:
        for i, line in enumerate(f):
            if i == 0:  # Skip the header line.
                continue
            coherent_1, coherent_2, incoherent_1, incoherent_2, _, _, _, _ = (
                line.rstrip('\n').split('\t'))
            # Strip because the second coherent sentence might be empty.
            fusion = (coherent_1 + ' ' + coherent_2).strip()
            yield [incoherent_1, incoherent_2], fusion


def read_label_map(path):
    """Returns label map read from the given path."""
    with tf.gfile.GFile(path) as f:
        if path.endswith('.json'):
            return json.load(f)
        else:
            label_map = {}
            empty_line_encountered = False
            for tag in f:
                tag = tag.strip()
                if tag:
                    label_map[tag] = len(label_map)
                else:
                    if empty_line_encountered:
                        raise ValueError(
                            'There should be no empty lines in the middle of the label map '
                            'file.'
                        )
                    empty_line_encountered = True
            return label_map
