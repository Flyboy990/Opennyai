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

# --- START OF COMPREHENSIVE MODIFICATION FOR HUGGINGFACE HUB COMPATIBILITY ---
# The original file used `pytorch_transformers.cached_path`.
# This has been updated to use the modern `huggingface_hub.hf_hub_download`.
# The `cached_download` function, which was a transition, has also been superseded by `hf_hub_download`.
#
# The custom CACHE_DIR from opennyai.utils.download is no longer used here.
# Hugging Face Hub manages its own default cache (`~/.cache/huggingface/hub`)
# or we can pass a `cache_dir` argument explicitly.
from huggingface_hub import hf_hub_download
from wasabi import msg
# --- END OF COMPREHENSIVE MODIFICATION ---


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
        return ids

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings) in a single string."""
        out_string = ""
        for (i, token) in enumerate(tokens):
            if i < len(tokens) - 1 and token.endswith("##"):
                out_string += token.replace("##", "")
            else:
                out_string += token + " "
        return out_string

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in a sequence of tokens (strings)."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    def convert_ids_to_string(self, ids):
        """Converts a sequence of ids in a single string."""
        return self.convert_tokens_to_string(self.convert_ids_to_tokens(ids))

    @classmethod
    def from_pretrained(cls, pretrained_vocab_name_or_path, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a BertTokenizer from a pre-trained vocabulary file.
        Download and cache the pre-trained vocabulary file if needed.
        """
        if cache_dir is None:
            # Use Hugging Face's default cache location if not specified
            # This is typically ~/.cache/huggingface/hub
            pass # No custom cache_dir needed if Hugging Face's default is acceptable
        
        if pretrained_vocab_name_or_path in PRETRAINED_VOCAB_ARCHIVE_MAP:
            vocab_url = PRETRAINED_VOCAB_ARCHIVE_MAP[pretrained_vocab_name_or_path]
            # --- START OF MODIFICATION ---
            # Corrected function call to hf_hub_download and parameter name
            # hf_hub_download uses 'url' and 'cache_dir' parameters
            vocab_file = hf_hub_download(url=vocab_url, cache_dir=cache_dir)
            # --- END OF MODIFICATION ---
        else:
            vocab_file = pretrained_vocab_name_or_path
        
        tokenizer = cls(vocab_file, *inputs, **kwargs)
        return tokenizer


class BasicTokenizer(object):
    """
    Runs basic tokenization (punctuation splitting, lower casing, etc.).
    """

    def __init__(self, do_lower_case=True, never_split=(
            "[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[unused0]", "[unused1]", "[unused2]", "[unused3]",
            "[unused4]", "[unused5]", "[unused6]")):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
          never_split: ("<unk>", "[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        orig_tokens = []
        for special_token in self.never_split:
            if special_token not in orig_tokens:
                orig_tokens.append(special_token)

        text = self._tokenize_chinese_chars(text)
        tokens = whitespace_tokenize(text)
        output_tokens = []
        for token in tokens:
            if self.do_lower_case:
                token = token.lower()
            output_tokens.extend(self._run_split_on_punc(token))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            if unicodedata.category(char) != "Mn":
                output.append(char)
        return "".join(output)

    def _run_clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            if ord(char) == 0 or ord(char) == 0xfffd or _is_control(char):
                continue
            output.append(char)
        return "".join(output)

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around Chinese characters."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or
                (cp >= 0x3400 and cp <= 0x4DBF) or
                (cp >= 0x20000 and cp <= 0x2A6DF) or
                (cp >= 0x2A700 and cp <= 0x2B73F) or
                (cp >= 0x2B740 and cp <= 0x2B81F) or
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or
                (cp >= 0x2F800 and cp <= 0x2FA1F)):
            return True

        return False

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

    def _is_punctuation(self, char):
        """Checks whether `chars` is a punctuation character."""
        cp = ord(char)
        if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
                (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
            return True
        if unicodedata.category(char).startswith("P"):
            return True
        return False


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its wordpiece tokens.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been lower cased and cleaned.

        Returns:
          A list of wordpiece tokens.
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
    if char == " " or char == "\t" or char == "\n" or char == "\r" or \
            char == "\f" or char == "\v":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    if char == "\t" or char == "\n" or char == "\r" or \
            char == "\f" or char == "\v":
        return True
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False