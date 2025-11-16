# -*- coding: utf-8 -*-
"""
Template pool for hard global constraints (description strings).

Super-categories (for your reference; keys本身保持兼容，不随分类变动):
1) length               -> digit_format_* / min_word_count
2) language             -> require_language_* / avoid_contractions
3) structure            -> has_sections_intro_body_conclusion / min_paragraphs
4) format_consistency   -> heading_levels_only / bullet_style_consistent / decimal_places / date_format_*
5) style_safety         -> forbid_first_person / forbid_emojis / symbol_format / keyword_format / citation_style_*

Step3 会对同一“超类”先收集候选，再随机选 1 条；同一键内也会随机抽候选句式。
Placeholders:
  {min_words}, {max_words}, {center}, {tol_pct}, {levels}, {marker},
  {keywords}, {places}, {symbol}
"""

from typing import Dict, List, Union
TemplateValue = Union[str, List[str]]

DESCS: Dict[str, TemplateValue] = {
    # ------------------------------
    # (1) length
    # ------------------------------
    "digit_format_min_max": [
        "Keep the answer length between {min_words} and {max_words} words.",
        "Target a response of {min_words}–{max_words} words.",
        "Constrain the total length to {min_words}–{max_words} words.",
        "Your answer should be no shorter than {min_words} and no longer than {max_words} words.",
        "Stay within {min_words}–{max_words} words overall.",
    ],
    "digit_format_around": [
        "Keep the answer length around {center} words (±{tol_pct}%).",
        "Aim for roughly {center} words, with a tolerance of ±{tol_pct}%.",
        "Produce about {center} words; deviations within ±{tol_pct}% are acceptable.",
        "Maintain an approximate length near {center} words (±{tol_pct}%).",
        "Center the response near {center} words, allowing ±{tol_pct}% variability.",
    ],
    "digit_format_min": [
        "The answer must be at least {min_words} words long.",
        "Ensure a minimum length of {min_words} words.",
        "Write no fewer than {min_words} words.",
        "Provide at least {min_words} words in total.",
        "Do not submit under {min_words} words.",
    ],
    "digit_format_max": [
        "Keep the answer under {max_words} words.",
        "Do not exceed {max_words} words.",
        "Limit the response to a maximum of {max_words} words.",
        "Cap the total length at {max_words} words.",
        "Stay below {max_words} words in length.",
    ],
    # 兼容性兜底：当未配置上述 digit_format_* 时使用
    "min_word_count": [
        "The answer must be at least {min_words} words long.",
        "Ensure a minimum length of {min_words} words.",
        "Provide a response of no less than {min_words} words.",
    ],

    # ------------------------------
    # (2) language
    # ------------------------------
    "require_language_en": [
        "The answer must be written primarily in English.",
        "Write the response mostly in English.",
        "Use English as the primary language for the answer.",
        "Ensure the response is predominantly in English.",
    ],
    "require_language_zh": [
        "The answer must be written primarily in Chinese.",
        "Write the response mostly in Chinese.",
        "Use Chinese as the primary language for the answer.",
        "Ensure the response is predominantly in Chinese.",
    ],
    "avoid_contractions": [
        "Avoid contractions (use 'do not' instead of 'don't').",
        "Do not use contractions; write out full forms.",
        "Write without contractions (e.g., 'cannot' instead of 'can't').",
        "Contractions are disallowed; prefer full phrases.",
        "Use formal forms rather than contractions throughout.",
    ],

    # ------------------------------
    # (3) structure
    # ------------------------------
    "has_sections_intro_body_conclusion": [
        "Include an Opening/Intro, a Body/Main Analysis, and a Conclusion/Outlook in logical progression.",
        "Structure the answer with an Introduction, a Main Body, and a Conclusion.",
        "Organize content into Intro → Body → Conclusion, maintaining a clear flow.",
        "Provide three parts—Intro, Body, Conclusion—with coherent transitions.",
        "Ensure the response has Intro, Main Analysis, and Conclusion sections.",
    ],
    "min_paragraphs": [
        "Organize the answer into at least {min_paras} paragraphs.",
        "Write a minimum of {min_paras} paragraphs.",
        "Present the response in no fewer than {min_paras} paragraphs.",
        "Break the content into {min_paras}+ paragraphs.",
        "Ensure at least {min_paras} distinct paragraphs.",
    ],

    # ------------------------------
    # (4) format_consistency
    # ------------------------------
    "heading_levels_only": [
        "Use consistent Markdown heading levels: only {levels}.",
        "Restrict headings to the following levels: {levels}.",
        "Maintain heading-level consistency; permitted levels are {levels}.",
        "Headings must only use levels {levels}.",
        "Keep heading hierarchy consistent; allow levels {levels} only.",
    ],
    "bullet_style_consistent": [
        "Use a consistent list marker style ('{marker}'); do not mix list markers.",
        "Keep bullet formatting uniform using '{marker}' only.",
        "Do not mix different bullet styles; stick to '{marker}'.",
        "Ensure lists consistently use the '{marker}' marker.",
        "Maintain a single bullet style across lists: '{marker}'.",
    ],
    "decimal_places": [
        "Keep numeric values to {places} decimal places consistently.",
        "Use {places}-decimal precision for all numbers.",
        "Format all numeric values with {places} decimal places.",
        "Maintain consistent {places}-place decimal formatting.",
        "Standardize numeric precision at {places} decimal places.",
    ],
    "date_format_iso": [
        "Use the date format YYYY-MM-DD.",
        "Dates must follow the ISO style: YYYY-MM-DD.",
        "Format dates as YYYY-MM-DD.",
        "Represent dates in ISO format (YYYY-MM-DD).",
        "Write dates strictly as YYYY-MM-DD.",
    ],
    "date_format_long": [
        "Use the date format 'Month DD, YYYY'.",
        "Format dates as Month DD, YYYY (e.g., March 14, 2024).",
        "Write dates in the long form: Month DD, YYYY.",
        "Represent dates as Month DD, YYYY (e.g., July 9, 2023).",
        "Use long-form dates like March 14, 2024.",
    ],

    # ------------------------------
    # (5) style_safety
    # ------------------------------
    "forbid_first_person": [
        "Maintain an objective, third-person analytic voice; do not use first-person pronouns.",
        "Avoid first-person narration; write in an objective, third-person tone.",
        "Use third-person analysis and refrain from 'I', 'we', 'my', or 'our'.",
        "Keep the register impersonal; avoid first-person references.",
        "Write without first-person perspective throughout.",
    ],
    "forbid_emojis": [
        "Do not use emojis or decorative unicode symbols.",
        "Avoid emojis and ornamental unicode characters.",
        "Refrain from including emojis or decorative symbols.",
        "Emojis and stylistic unicode characters are not allowed.",
        "Exclude emojis or decorative glyphs from the response.",
    ],
    "symbol_format": [
        "Do not use the symbol '{symbol}'.",
        "Avoid the symbol '{symbol}' in the response.",
        "Refrain from using '{symbol}' anywhere in the text.",
        "The character '{symbol}' must not appear in the answer.",
        "Prohibit the use of '{symbol}' throughout.",
    ],
    "keyword_format": [
        "Include the following keywords: {keywords}.",
        "Ensure the response explicitly contains these keywords: {keywords}.",
        "The answer must mention the following keywords: {keywords}.",
        "Use these keywords verbatim somewhere in the text: {keywords}.",
        "Incorporate the keywords exactly as listed: {keywords}.",
    ],
    "citation_style_numeric": [
        "Use numeric bracket citations like [1], [2].",
        "Cite sources with numeric brackets (e.g., [1], [2]).",
        "Apply the numeric bracket citation style: [n].",
        "Use bracketed numeric references, such as [3].",
        "Format citations as bracketed numbers, e.g., [4].",
    ],
    "citation_style_author_year": [
        "Use author–year citations like (Smith, 2021).",
        "Cite sources in the author–year format (e.g., (Lee, 2020)).",
        "Apply the author–year citation style: (Author, YYYY).",
        "Use parenthetical author–year references, e.g., (Garcia, 2019).",
        "Format citations in author–year style such as (Kim, 2022).",
    ],
}