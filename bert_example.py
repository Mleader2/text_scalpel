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

"""Build BERT Examples from text (source, target) pairs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import tagging
import tensorflow as tf
from utils import my_tokenizer_class
from curLine_file import curLine

class BertExample(object):
    """Class for training and inference examples for BERT.

    Attributes:
      editing_task: The EditingTask from which this example was created. Needed
        when realizing labels predicted for this example.
      features: Feature dictionary.
    """

    def __init__(self, input_ids,
                 input_mask,
                 segment_ids, labels,
                 labels_mask,
                 token_start_indices,
                 task, default_label):
        input_len = len(input_ids)
        if not (input_len == len(input_mask) and input_len == len(segment_ids) and
                input_len == len(labels) and input_len == len(labels_mask)):
            raise ValueError(
                'All feature lists should have the same length ({})'.format(
                    input_len))

        self.features = collections.OrderedDict([
            ('input_ids', input_ids),
            ('input_mask', input_mask),
            ('segment_ids', segment_ids),
            ('labels', labels),
            ('labels_mask', labels_mask),
        ])
        self._token_start_indices = token_start_indices
        self.editing_task = task
        self._default_label = default_label

    def pad_to_max_length(self, max_seq_length, pad_token_id):
        """Pad the feature vectors so that they all have max_seq_length.

        Args:
          max_seq_length: The length that features will have after padding.
          pad_token_id: input_ids feature is padded with this ID, other features
            with ID 0.
        """
        pad_len = max_seq_length - len(self.features['input_ids'])
        for key in self.features:
            pad_id = pad_token_id if key == 'input_ids' else 0
            self.features[key].extend([pad_id] * pad_len)
            if len(self.features[key]) != max_seq_length:
                raise ValueError('{} has length {} (should be {}).'.format(
                    key, len(self.features[key]), max_seq_length))

    def to_tf_example(self):
        """Returns this object as a tf.Example."""

        def int_feature(values):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

        tf_features = collections.OrderedDict([
            (key, int_feature(val)) for key, val in self.features.items()
        ])
        return tf.train.Example(features=tf.train.Features(feature=tf_features))

    def get_token_labels(self):
        """Returns labels/tags for the original tokens, not for wordpieces."""
        labels = []
        for idx in self._token_start_indices:
            # For unmasked and untruncated tokens, use the label in the features, and
            # for the truncated tokens, use the default label.
            if (idx < len(self.features['labels']) and
                    self.features['labels_mask'][idx]):
                current_label = self.features['labels'][idx]
                # if current_label >= 0:
                labels.append(self.features['labels'][idx])
                # else:  # stop
                #     labels.append(self._default_label)
            else:
                labels.append(self._default_label)
            if labels[-1]<0:
                print(curLine(), idx, len(self.features['labels']), "mask=",self.features['labels_mask'][idx], self.features['labels'][idx], labels[-1] )
        return labels


class BertExampleBuilder(object):
    """Builder class for BertExample objects."""

    def __init__(self, label_map, vocab_file,
                 max_seq_length, do_lower_case,
                 converter):
        """Initializes an instance of BertExampleBuilder.

        Args:
          label_map: Mapping from tags to tag IDs.
          vocab_file: Path to BERT vocabulary file.
          max_seq_length: Maximum sequence length.
          do_lower_case: Whether to lower case the input text. Should be True for
            uncased models and False for cased models.
          converter: Converter from text targets to tags.
        """
        self._label_map = label_map
        self._tokenizer = my_tokenizer_class(vocab_file, do_lower_case=do_lower_case)
        self._max_seq_length = max_seq_length
        self._converter = converter
        self._pad_id = self._get_pad_id()
        self._keep_tag_id = self._label_map['KEEP']

    def build_bert_example(
            self,
            sources,
            target=None,
            use_arbitrary_target_ids_for_infeasible_examples=False,
            location=None
    ):
        """Constructs a BERT Example.

        Args:
          sources: List of source texts.
          target: Target text or None when building an example during inference.
          use_arbitrary_target_ids_for_infeasible_examples: Whether to build an
            example with arbitrary target ids even if the target can't be obtained
            via tagging.

        Returns:
          BertExample, or None if the conversion from text to tags was infeasible
          and use_arbitrary_target_ids_for_infeasible_examples == False.
        """
        # Compute target labels.
        task = tagging.EditingTask(sources, location=location, tokenizer=self._tokenizer)
        if target is not None:
            tags = self._converter.compute_tags(task, target, tokenizer=self._tokenizer)
            if not tags:  # 不可转化，取决于　use_arbitrary_target_ids_for_infeasible_examples
                if use_arbitrary_target_ids_for_infeasible_examples:
                    # Create a tag sequence [KEEP, DELETE, KEEP, DELETE, ...] which is
                    # unlikely to be predicted by chance.
                    tags = [tagging.Tag('KEEP') if i % 2 == 0 else tagging.Tag('DELETE')
                            for i, _ in enumerate(task.source_tokens)]
                else:
                    return None
        else:
            # If target is not provided, we set all target labels to KEEP.
            tags = [tagging.Tag('KEEP') for _ in task.source_tokens]
        labels = [self._label_map[str(tag)] for tag in tags]
        # tokens, labels, token_start_indices = self._split_to_wordpieces( #  wordpiece： tag是以ｗｏｒｄ为单位的，组成ｗｏｒｄ的ｐｉｅｃｅ的标注与这个ｗｏｒｄ相同
        #     task.source_tokens, labels)
        if len(task.source_tokens) > self._max_seq_length - 2:
            print(curLine(), "%d tokens is to long," % len(task.source_tokens), "truncate task.source_tokens:",
                  task.source_tokens)
        token_start_indices = [indices+1 for indices in range(len(task.source_tokens))]

        #  截断到self._max_seq_length - 2
        tokens = self._truncate_list(task.source_tokens)
        labels = self._truncate_list(labels)

        input_tokens = ['[CLS]'] + tokens + ['[SEP]']
        labels_mask = [0] + [1] * len(labels) + [0]
        labels = [0] + labels + [0]

        input_ids = self._tokenizer.convert_tokens_to_ids(input_tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        example = BertExample(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            labels=labels,
            labels_mask=labels_mask,
            token_start_indices=token_start_indices,
            task=task,
            default_label=self._keep_tag_id)
        example.pad_to_max_length(self._max_seq_length, self._pad_id)
        return example

    # def _split_to_wordpieces(self, tokens, labels):
    #   """Splits tokens (and the labels accordingly) to WordPieces.
    #
    #   Args:
    #     tokens: Tokens to be split.
    #     labels: Labels (one per token) to be split.
    #
    #   Returns:
    #     3-tuple with the split tokens, split labels, and the indices of the
    #     WordPieces that start a token.
    #   """
    #   bert_tokens = []  # Original tokens split into wordpieces.
    #   bert_labels = []  # Label for each wordpiece.
    #   # Index of each wordpiece that starts a new token.
    #   token_start_indices = []
    #   for i, token in enumerate(tokens):
    #     # '+ 1' is because bert_tokens will be prepended by [CLS] token later.
    #     token_start_indices.append(len(bert_tokens) + 1)
    #     pieces = self._tokenizer.tokenize(token)
    #     bert_tokens.extend(pieces)
    #     bert_labels.extend([labels[i]] * len(pieces))
    #   return bert_tokens, bert_labels, token_start_indices

    def _truncate_list(self, x):
        """Returns truncated version of x according to the self._max_seq_length."""
        # Save two slots for the first [CLS] token and the last [SEP] token.
        return x[:self._max_seq_length - 2]

    def _get_pad_id(self):
        """Returns the ID of the [PAD] token (or 0 if it's not in the vocab)."""
        try:
            return self._tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
        except KeyError:
            return 0
