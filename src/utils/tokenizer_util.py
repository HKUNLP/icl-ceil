#!/usr/bin/python3
# -*- coding: utf-8 -*-
from transformers import AutoTokenizer


def model_to_tokenizer(model_name):
    if "code-" in model_name:
        return "SaulLu/codex-like-tokenizer"
    if "gpt3" in model_name:
        return "gpt2"
    return model_name


def get_tokenizer(model_name):
    if model_name == 'bm25':
        return model_name
    return AutoTokenizer.from_pretrained(model_to_tokenizer(model_name))