from __future__ import annotations

import emoji
import json
import nltk
import re
import logging
import collections
import string
import functools
from typing import Any, Dict, Optional, List, Iterable

# Try importing langdetect, handle if missing
try:
    import langdetect
    from langdetect import detect, LangDetectException
except ImportError:
    langdetect = None
    detect = None
    LangDetectException = Exception

# Try importing pythainlp, handle if missing
try:
    from pythainlp.tokenize import sent_tokenize as sent_tokenize_thai
    from pythainlp.tokenize import word_tokenize as word_tokenize_thai
except ImportError:
    sent_tokenize_thai = None
    word_tokenize_thai = None

from .base import BaseMetric

logger = logging.getLogger(__name__)

# Constants from ifeval.py
_CHINESE_CHARS_PATTERN = r"[\u4E00-\u9FFF\u3400-\u4DBF]"
_JAPANESE_CHARS_PATTERN = r"[\u3040-\u309f\u30a0-\u30ff]"
_KOREAN_CHARS_PATTERN = r"[\uAC00-\uD7AF]"

def split_chinese_japanese(lines: str) -> Iterable[str]:
    """Split Chinese and Japanese text into sentences."""
    for line in lines.splitlines():
        for sent in re.findall(
            r"[^!?。\.\!\?\！\？\．\n]+[!?。\.\!\?\！\？\．\n]?", line.strip(), flags=re.U
        ):
            yield sent

def count_words_chinese_japanese(text: str) -> int:
    """Counts the number of words for Chinese, Japanese and Korean."""
    non_alphanumeric_patterns = (
        r"[\\.\!\?\．\/_,\{\}<>:;$%^&*(+\"\'+——！，。？、`~@#￥……（）：；《）《》“”()\[\]»〔〕\-「」]+"
    )
    text = re.sub(non_alphanumeric_patterns, "", text)
    if emoji:
        emoji_cnt = emoji.emoji_count(text)
        text = emoji.replace_emoji(text, "")
    else:
        emoji_cnt = 0
    foreign_chars_patterns = "|".join(
        [_CHINESE_CHARS_PATTERN, _JAPANESE_CHARS_PATTERN, _KOREAN_CHARS_PATTERN]
    )
    asian_chars = re.findall(foreign_chars_patterns, text)
    asian_chars_cnt = len(asian_chars)
    non_asian_chars = re.sub(foreign_chars_patterns, " ", text)
    non_asian_words_cnt = len(non_asian_chars.split())
    return non_asian_words_cnt + asian_chars_cnt + emoji_cnt

def count_hindi_num_sentences(text):
    sentences = re.split(r'(?<=[।!?])\s*', text)
    return len([s for s in sentences if s.strip()])

@functools.lru_cache(maxsize=None)
def _get_sentence_tokenizer():
    return nltk.data.load("nltk:tokenizers/punkt/english.pickle")

def count_sentences(text):
    """Count the number of sentences (English/Default)."""
    tokenizer = _get_sentence_tokenizer()
    tokenized_sentences = tokenizer.tokenize(text)
    return len(tokenized_sentences)

def count_words(text):
    """Counts the number of words (English/Default)."""
    try:
        tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
        tokens = tokenizer.tokenize(text)
        num_words = len(tokens)
    except:
        return 0
    return num_words

class InstructionFollowingMetric(BaseMetric):
    
    
    def compute(
        self,
        prediction: str,
        reference: Optional[str],
        history_messages: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Computes if the prediction follows the specified instruction.
        
        kwargs expected to contain:
        - inst_name: str (The key name of the instruction)
        - inst_args: Dict[str, Any] (Arguments for the instruction check)
        """
        inst_name = kwargs.get("inst_name")
        inst_args = kwargs.get("inst_args", {})
        
        if not inst_name:
            return {"score": 0.0, "rationale": "Missing instruction name."}

        handler_map = {
            "startend:start_char": self._check_start_char,
            "startend:start_emoji": self._check_start_emoji,
            "startend:end_phrase": self._check_end_phrase,
            "language:response_language": self._check_response_language,
            "format:json_format": self._check_json_format,
            "format:bullet_list": self._check_bullet_list,
            "length_constraints:number_sentences": self._check_number_sentences,
            "keywords:existence": self._check_keyword_existence,
            "change_case:capital_letter": self._check_capital_letter,
            "change_case:lowercase": self._check_lowercase,
            "punctuation:no_comma": self._check_comma,
            "combination:two_responses": self._check_two_responses,
            "content:placeholder": self._check_placeholder,
            "format:constrained_response": self._check_constrained_response,
            "keywords:frequency": self._check_keyword_frequency,
            "keywords:forbidden_words": self._check_forbidden_words,
            "keywords:letter_frequency": self._check_letter_frequency,
            "length_constraints:number_paragraphs": self._check_number_paragraphs,
            "length_constraints:number_words": self._check_number_words,
            "length_constraints:nth_paragraph_first_word": self._check_nth_paragraph_first_word,
            "content:postscript": self._check_postscript,
            "format:number_highlighted_sections": self._check_highlighted_sections,
            "format:multiple_sections": self._check_multiple_sections,
            "format:title": self._check_title,
            "combination:repeat_prompt": self._check_repeat_prompt,
            "change_case:capital_word_frequency": self._check_capital_word_frequency,
            "startend:quotation": self._check_quotation,
        }
        
        handler = handler_map.get(inst_name)
        if not handler:
            return {"score": 0.0, "rationale": f"Unknown instruction: {inst_name}"}
            
        # prediction is the response to verify
        if prediction is None:
            prediction = ""
            
        try:
            is_following = handler(prediction, inst_args)
        except Exception as e:
            logger.error(f"Error checking instruction {inst_name}: {e}")
            is_following = False
        
        return {
            "score": 1.0 if is_following else 0.0,
            "rationale": "Followed" if is_following else "Not followed"
        }

    def _check_start_char(self, response: str, args: Dict) -> bool:
        letter = args.get("letter", "")
        if not letter:
             return False
        return response.strip().lower().startswith(letter.lower())

    def _check_start_emoji(self, response: str, args: Dict) -> bool:
        response = response.strip()
        if not response:
            return False
        return emoji.is_emoji(response[0])

    def _check_end_phrase(self, response: str, args: Dict) -> bool:
        end_phrase = args.get("end_phrase")
        if not end_phrase:
            return False
        # ifeval logic: strip double quotes
        response = response.strip().strip('"').lower()
        return response.endswith(end_phrase.lower())

    def _check_response_language(self, response: str, args: Dict) -> bool:
        if langdetect is None:
            logger.warning("langdetect not installed, skipping language check.")
            return False

        langs = {
            "English": "en", "Spanish": "es", "Portuguese": "pt", "Arabic": "ar",
            "Hindi": "hi", "French": "fr", "Russian": "ru", "German": "de",
            "Japanese": "ja", "Italian": "it", "Bengali": "bn", "Ukrainian": "uk",
            "Thai": "th", "Urdu": "ur", "Tamil": "ta", "Telugu": "te",
            "Bulgarian": "bg", "Korean": "ko", "Polish": "pl", "Hebrew": "he",
            "Persian": "fa", "Vietnamese": "vi", "Nepali": "ne", "Swahili": "sw",
            "Kannada": "kn", "Marathi": "mr", "Gujarati": "gu", "Punjabi": "pa",
            "Malayalam": "ml", "Finnish": "fi",
        }
        lang_key = args.get("lang")
        if not lang_key or lang_key not in langs:
            return False
            
        target_lang_code = langs[lang_key]
        
        response = response.strip()
        if not response:
            return False
        try:
            langdetect.detect(response)
        except LangDetectException:
            # According to original implementation, return True on exception
            return True
            
        return detect(response) == target_lang_code

    def _check_json_format(self, response: str, args: Dict) -> bool:
        response = (
            response.strip()
            .removeprefix("```json")
            .removeprefix("```Json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            json.loads(response)
        except ValueError:
            return False
        return True

    def _check_bullet_list(self, response: str, args: Dict) -> bool:
        num_bullets = args.get("num_bullets")
        if num_bullets is None:
            return False

        bullet_lists = re.findall(r"^\s*\*[^\*].*$", response, flags=re.MULTILINE)
        bullet_lists_2 = re.findall(r"^\s*-.*$", response, flags=re.MULTILINE)
        num_bullet_lists = len(bullet_lists) + len(bullet_lists_2)
        
        if num_bullets == 0:
            return num_bullet_lists > 0
        return num_bullet_lists == num_bullets

    def _check_number_sentences(self, response: str, args: Dict) -> bool:
        num_sentences = args.get("num_sentences")
        relation = args.get("relation")
        
        if num_sentences is None or not relation:
            return False
            
        # Language-aware sentence counting
        try:
            lang = langdetect.detect(response) if langdetect else 'en'
        except:
            lang = 'en'
            
        if lang == "th" and sent_tokenize_thai:
            actual_sentences = sum(
                [len(sent_tokenize_thai(line)) for line in response.splitlines()]
            )
        elif lang == 'hi':
            actual_sentences = count_hindi_num_sentences(response)
        elif lang in ["zh", "zh-cn", "zh-tw", "ja"]:
            actual_sentences = len(list(split_chinese_japanese(response)))
        else:
            actual_sentences = count_sentences(response)
        
        if relation == "less than":
            return actual_sentences < num_sentences
        elif relation == "at least" or relation == "more than":
            # ifeval uses "at least" (>=), we map "more than" to it or handle it
            if relation == "at least":
                return actual_sentences >= num_sentences
            return actual_sentences > num_sentences
        return False

    def _check_keyword_existence(self, response: str, args: Dict) -> bool:
        keywords = args.get("keywords", [])
        for keyword in keywords:
            # ifeval uses re.search with IGNORECASE
            if not re.search(re.escape(keyword), response, flags=re.IGNORECASE):
                return False
        return True

    def _check_capital_letter(self, response: str, args: Dict) -> bool:
        # ifeval checks for english
        if langdetect:
            try:
                if langdetect.detect(response) != "en":
                    return False
            except:
                pass 
        return response.isupper()

    def _check_lowercase(self, response: str, args: Dict) -> bool:
         # ifeval checks for english
        if langdetect:
            try:
                if langdetect.detect(response) != "en":
                     return False
            except:
                pass
        return response.islower()

    def _check_comma(self, response: str, args: Dict) -> bool:
        return "," not in response

    def _check_two_responses(self, response: str, args: Dict) -> bool:
        valid_responses = list()
        responses = response.split("******")
        for index, res in enumerate(responses):
            if not res.strip():
                # Might be empty spaces in the beginning or the end.
                if index != 0 and index != len(responses) - 1:
                    return False
            else:
                valid_responses.append(res)
        return (
            len(valid_responses) == 2
            and valid_responses[0].strip() != valid_responses[1].strip()
        )

    def _check_placeholder(self, response: str, args: Dict) -> bool:
        num_placeholder = args.get("num_placeholder")
        if num_placeholder is None:
            return False
            
        placeholders = re.findall(r"\[.*?\]", response)
        num_placeholders = len(placeholders)
        return num_placeholders >= num_placeholder

    def _check_constrained_response(self, response: str, args: Dict) -> bool:
        response_options = args.get("response_options", [])
        response = response.strip() # ifeval strips
        for const in response_options:
            if const in response:
                return True
        return False

    def _check_keyword_frequency(self, response: str, args: Dict) -> bool:
        keyword = args.get("keyword")
        frequency = args.get("frequency")
        relation = args.get("relation")

        if not keyword or frequency is None or not relation:
            return False

        try:
            actual_occurrences = len(re.findall(keyword, response, flags=re.IGNORECASE))
        except:
            return False

        if relation == "less than":
            return actual_occurrences < frequency
        elif relation == "at least":
            return actual_occurrences >= frequency
        return False

    def _check_forbidden_words(self, response: str, args: Dict) -> bool:
        forbidden_words = args.get("forbidden_words", [])
        for word in forbidden_words:
            if re.search(r"\b" + re.escape(word) + r"\b", response, flags=re.IGNORECASE):
                return False
        return True

    def _check_letter_frequency(self, response: str, args: Dict) -> bool:
        letter = args.get("letter")
        frequency = args.get("let_frequency")
        relation = args.get("let_relation")

        if not letter or frequency is None or not relation:
            return False

        response = response.lower()
        letters = collections.Counter(response)
        count = letters.get(letter.lower(), 0)

        if relation == "less than":
            return count < frequency
        elif relation == "at least":
            return count >= frequency
        return False

    def _check_number_paragraphs(self, response: str, args: Dict) -> bool:
        num_paragraphs = args.get("num_paragraphs")
        if num_paragraphs is None:
            return False

        paragraphs = re.split(r"\s?\*\*\*\s?", response)
        actual_num = len(paragraphs)

        for index, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                if index == 0 or index == len(paragraphs) - 1:
                    actual_num -= 1
                else:
                    return False

        return actual_num == num_paragraphs

    def _check_number_words(self, response: str, args: Dict) -> bool:
        num_words = args.get("num_words")
        relation = args.get("relation")

        if num_words is None or not relation:
            return False

        # Language-aware word counting
        try:
            lang = langdetect.detect(response) if langdetect else 'en'
        except:
            lang = 'en'
            
        if lang == "th" and word_tokenize_thai:
            actual_words = len(word_tokenize_thai(response))
        elif lang in ["zh", "zh-cn", "zh-tw", "ja"]:
            actual_words = count_words_chinese_japanese(response)
        else:
            actual_words = count_words(response)

        if relation == "less than":
            return actual_words < num_words
        elif relation == "at least":
            return actual_words >= num_words
        elif relation == "more than":
            return actual_words > num_words
        return False

    def _check_nth_paragraph_first_word(self, response: str, args: Dict) -> bool:
        num_paragraphs = args.get("num_paragraphs")
        nth_paragraph = args.get("nth_paragraph")
        first_word = args.get("first_word")

        if num_paragraphs is None or nth_paragraph is None or not first_word:
            return False

        paragraphs = re.split(r"\n\n", response)
        
        actual_paragraphs = []
        for p in paragraphs:
            if p.strip():
                actual_paragraphs.append(p)
                
        if len(actual_paragraphs) != num_paragraphs:
            return False
            
        if nth_paragraph > len(actual_paragraphs) or nth_paragraph < 1:
            return False

        target_paragraph = actual_paragraphs[nth_paragraph - 1].strip()
        if not target_paragraph:
            return False

        word_parts = target_paragraph.split()
        if not word_parts:
            return False
            
        word = word_parts[0].strip()
        word = word.lstrip("'").lstrip('"')

        actual_first_word = ""
        punctuation = {".", ",", "?", "!", "'", '"'}
        for char in word:
            if char in punctuation:
                break
            actual_first_word += char.lower()

        return actual_first_word == first_word.lower()

    def _check_postscript(self, response: str, args: Dict) -> bool:
        postscript_marker = args.get("postscript_marker")
        if not postscript_marker:
            return False

        response = response.lower()
        marker = postscript_marker.lower()

        if marker == "p.p.s":
            pattern = r"\s*p\.\s?p\.\s?s.*$"
        elif marker == "p.s.":
            pattern = r"\s*p\.\s?s\..*$"
        else:
            pattern = r"\s*" + re.escape(marker) + r".*$"

        return bool(re.findall(pattern, response, flags=re.MULTILINE))

    def _check_highlighted_sections(self, response: str, args: Dict) -> bool:
        num_highlights = args.get("num_highlights")
        if num_highlights is None:
            return False

        count = 0
        try:
            highlights = re.findall(r"\*[^\n\*]*\*", response)
        except:
            return False
            
        double_highlights = re.findall(r"\*\*[^\n\*]*\*\*", response)

        for highlight in highlights:
            if highlight.strip("*").strip():
                count += 1
        for highlight in double_highlights:
             if highlight.replace("**", "").strip():
                 count += 1

        return count >= num_highlights

    def _check_multiple_sections(self, response: str, args: Dict) -> bool:
        section_spliter = args.get("section_spliter")
        num_sections = args.get("num_sections")

        if not section_spliter or num_sections is None:
            return False

        pattern = r"\s?" + section_spliter + r"\s?\d+\s?"
        try:
            sections = re.split(pattern, response)
        except:
             pattern = r"\s?" + re.escape(section_spliter) + r"\s?\d+\s?"
             sections = re.split(pattern, response)
             
        actual_sections = len(sections) - 1

        return actual_sections >= num_sections

    def _check_title(self, response: str, args: Dict) -> bool:
        pattern = r"<<[^\n]+>>"
        titles = re.findall(pattern, response)
        for title in titles:
            if title.lstrip("<").rstrip(">").strip():
                return True
        return False

    def _check_repeat_prompt(self, response: str, args: Dict) -> bool:
        prompt_to_repeat = args.get("prompt_to_repeat")
        if not prompt_to_repeat:
            return False
        return response.strip().lower().startswith(prompt_to_repeat.strip().lower())

    def _check_capital_word_frequency(self, response: str, args: Dict) -> bool:
        frequency = args.get("capital_frequency")
        relation = args.get("capital_relation")

        if frequency is None or not relation:
            return False

        words = nltk.word_tokenize(response)
        capital_words = [word for word in words if word.isupper()]
        count = len(capital_words)

        if relation == "less than":
            return count < frequency
        elif relation == "at least":
            return count >= frequency
        return False

    def _check_quotation(self, response: str, args: Dict) -> bool:
        value = response.strip()
        if len(value) < 2:
            return False
        return (value[0] == '"' and value[-1] == '"') or \
               (value[0] == '“' and value[-1] == '”') or \
               (value[0] == '「' and value[-1] == '」')
