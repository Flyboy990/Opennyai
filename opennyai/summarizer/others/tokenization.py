# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""Tokenization classes."""

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import os
import unicodedata
from io import open

# --- START OF MODIFICATION ---
# Replaced 'from transformers import cached_path' with the modern equivalent from huggingface_hub
from huggingface_hub import cached_download
# --- END OF MODIFICATION ---
from wasabi import msg

from opennyai.utils.download import CACHED_DIR # Corrected import to CACHED_DIR

# --- START OF MODIFICATION ---
# If CACHED_DIR is a local path, then cached_path was essentially a no-op here for resolution.
# We assume the intent is simply to define the local path to the summarizer model file.
EXTRACTIVE_SUMMARIZER_CACHE_PATH = os.path.join(CACHED_DIR, 'ExtractiveSummarizer')
# --- END OF MODIFICATION ---

PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
}
PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP = {
    'bert-base-uncased': 512,
    'bert-large-uncased': 512,
    'bert-base-cased': 512,
    'bert-large-cased': 512,
    'bert-base-multilingual-uncased': 512,
    'bert-base-multilingual-cased': 512,
    'bert-base-chinese': 512,
}
VOCAB_NAME = 'vocab.txt'


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BertTokenizer(object):
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""

    def __init__(self, vocab_file, do_lower_case=True, max_len=None,
                 never_split=(
                         "[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[unused0]", "[unused1]", "[unused2]",
                         "[unused3]",
                         "[unused4]", "[unused5]", "[unused6]")):

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))
        self.do_lower_case = do_lower_case
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
                                              never_split=never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1e12)

    def tokenize(self, text, use_bert_basic_tokenizer=False):
        split_tokens = []
        if (use_bert_basic_tokenizer):
            pretokens = self.basic_tokenizer.tokenize(text)
        else:
            pretokens = list(enumerate(text.split()))

        for i, token in pretokens:
            # if(self.do_lower_case):
            #     token = token.lower()
            subtokens = self.wordpiece_tokenizer.tokenize(token)
            for sub_token in subtokens:
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        # if len(ids) > self.max_len:
        #     raise ValueError(
        #         "Token indices sequence length is longer than the specified maximum "
        #         " sequence length for this BERT model ({} > {}). "
        #         "Run this tokenizer with `max_length` to prevent this error.".format(len(ids), self.max_len))
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids (integers) into tokens (strings) using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-word tokenization usually)
        in a single string."""
        out_string = "".join(tokens).replace("##", "").strip()
        return out_string

    def save_vocabulary(self, save_directory):
        """Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (:obj:`str`):
                The directory in which to save the vocabulary.
        """
        if os.path.isdir(save_directory):
            vocabulary_path = os.path.join(save_directory, VOCAB_NAME)
            with open(vocabulary_path, "w", encoding="utf-8") as writer:
                for token, index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                    writer.write(token + "\n")
        return [vocabulary_path]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=CACHED_DIR, *inputs,
                        **kwargs): # Corrected default to CACHED_DIR

        if pretrained_model_name_or_path in PRETRAINED_VOCAB_ARCHIVE_MAP:
            vocab_file = PRETRAINED_VOCAB_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            vocab_file = pretrained_model_name_or_path
        if os.path.isdir(vocab_file):
            vocab_file = os.path.join(vocab_file, VOCAB_NAME)

        # --- START OF MODIFICATION ---
        # Replaced cached_path with huggingface_hub.cached_download
        # This handles remote URLs and ensures files are cached locally.
        try:
            # Check if it's a URL first; cached_download is best for URLs.
            # If it's a local path, it should already be available.
            if vocab_file.startswith('http://') or vocab_file.startswith('https://'):
                # cached_download requires a `url` argument
                resolved_vocab_file = cached_download(url=vocab_file, local_dir=cache_dir)
            else:
                # If it's not a URL, assume it's a local file path
                # and ensure it exists, similar to cached_path's local behavior.
                resolved_vocab_file = vocab_file
                if not os.path.exists(resolved_vocab_file):
                     raise EnvironmentError(f"Local vocab file not found at {resolved_vocab_file}")

        except EnvironmentError:
        # --- END OF MODIFICATION ---
            msg.fail(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_VOCAB_ARCHIVE_MAP.keys()),
                    vocab_file))
            return None

        for key in kwargs:
            if key in ["do_lower_case", "col_sep", "never_split", "pad_token",
                       "unk_token", "cls_token", "sep_token", "mask_token",
                       "build_inputs_with_special_tokens_func", "max_len",
                       "init_kwargs"]:
                if key == "max_len":
                    continue
                kwargs.pop(key)

        return cls(resolved_vocab_file, *inputs, **kwargs)


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[unused0]", "[unused1]", "[unused2]",
                               "[unused3]",
                               "[unused4]", "[unused5]", "[unused6]")):
        """Constructs a BasicTokenizer.

        Args:
            do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        # This is a simple BPM method which is sufficient for simple texts. For
        # more complicated texts with many word containing chars for different
        # languages like Chinese/Japanese chars with space it is recommended to use
        # some of the publicly available sub-word tokenizers that support these
        # texts.
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            if unicodedata.category(char) != "Mn":
                output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            if char == "" or char == "\n" or char == "\r" or char == "\t":
                continue
            if _is_control(char):
                continue
            output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its wordpiece sub-tokens.

        This is the original code from Google's modeling code.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are technically printed.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we treat them as whitespace
    # by default in this script.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
