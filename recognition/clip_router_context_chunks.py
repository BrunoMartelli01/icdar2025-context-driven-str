"""Validated text chunks for the approximate CLIP literature experiment."""
from __future__ import annotations


POLICY = 'validated_whitespace_boundaries_v1'


def prompt_chunks(template, text, tokenizer, limit):
    """Preserve raw characters and validate each preferred whitespace boundary.

    Token lengths need not increase monotonically with character prefixes. A
    space can also be the first character beyond the longest fitting prefix.
    Only boundaries whose complete formatted prompt fits may be selected.
    """
    if template.count('{text}') != 1 or limit < 1:
        raise ValueError('Invalid prompt/context limit')

    def size(value):
        return len(tokenizer(template.format(text=value),
            add_special_tokens=True, truncation=False)['input_ids'])

    if size('') >= limit:
        raise ValueError('Prompt prefix exhausts context')
    if not text:
        return [template.format(text='')]
    parts, remaining = [], text
    while remaining:
        if size(remaining) <= limit:
            parts.append(remaining)
            break
        valid = [i for i in range(1, len(remaining) + 1)
                 if size(remaining[:i]) <= limit]
        if not valid:
            raise ValueError('A character cannot fit in model context')
        spaces = [i for i in valid if i > 1 and remaining[i - 1] == ' ']
        end = max(spaces or valid)
        parts.append(remaining[:end])
        remaining = remaining[end:]
    if ''.join(parts) != text:
        raise AssertionError('Chunking lost transcript characters')
    prompts = [template.format(text=part) for part in parts]
    if any(len(tokenizer(prompt, add_special_tokens=True,
           truncation=False)['input_ids']) > limit for prompt in prompts):
        raise AssertionError('Validated prompt chunk exceeds context')
    return prompts
