"""DSPy program (brevity-constrained CoT) and metric for DTM multiple-choice answering."""
from __future__ import annotations

import functools
import re
import dspy

CHOICE_LETTERS = "ABCD"


@functools.lru_cache(maxsize=None)
def _boundary_letter_re(letters: str) -> re.Pattern:
    return re.compile(rf"\b([{letters}])\b")


@functools.lru_cache(maxsize=None)
def _adapter_answer_re(letters: str) -> re.Pattern:
    return re.compile(rf"answer_letter\s*#*\s*\]\]\s*\(?\s*([{letters}])\b", re.I)


def parse_letter(text: str, letters: str = CHOICE_LETTERS) -> str:
    """Extract the answer letter (within `letters`) from model output."""
    text = (text or "").strip()
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip() or text
    matches = list(_boundary_letter_re(letters).finditer(cleaned.upper()))
    if matches:
        return matches[-1].group(1)
    if cleaned and cleaned[0].upper() in letters:
        return cleaned[0].upper()
    return cleaned[:1].upper() if cleaned else ""


class ConciseCoT(dspy.Signature):
    """O'zbek tilidagi test savoliga javob ber."""  # Answer the Uzbek test question.
    question: str = dspy.InputField()
    options: str = dspy.InputField()
    reasoning: str = dspy.OutputField(desc="Juda qisqa mulohaza, ko'pi bilan 2 gap.")
    answer_letter: str = dspy.OutputField(desc="Faqat bitta harf: A, B, C yoki D.")


class CoTSolver(dspy.Module):
    """Brevity-constrained zero-shot chain of thought (reasoning -> letter)."""
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ConciseCoT)

    def forward(self, question, options):
        r = self.predict(question=question, options=options)
        return dspy.Prediction(answer_letter=r.answer_letter,
                               reasoning=getattr(r, "reasoning", ""))


def metric(example, pred, trace=None, letters: str = CHOICE_LETTERS) -> bool:
    """Exact-match on the answer letter."""
    return parse_letter(pred.answer_letter, letters) == str(example.answer_letter).strip().upper()


SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)")
ADAPTER_ANSWER = _adapter_answer_re(CHOICE_LETTERS)  # back-compat alias, same pattern as before


def is_compliant(reasoning: str, max_sentences: int = 2, max_chars: int = 300) -> bool:
    """Does the reasoning respect the signature's brevity instruction
    ('Juda qisqa mulohaza, ko'pi bilan 2 gap')?"""
    r = (reasoning or "").strip()
    return len(SENTENCE_END.findall(r)) <= max_sentences or len(r) <= max_chars


def compliant_metric(example, pred, trace=None, letters: str = CHOICE_LETTERS) -> bool:
    """Correct AND instruction-compliant: the one-line fix (spec E5a)."""
    return metric(example, pred, trace, letters) and is_compliant(getattr(pred, "reasoning", ""))


def demo_field(demo, name: str, default: str = "") -> str:
    """Read a field from a demonstration, whichever form it is in.

    Demos we construct are `dspy.Example` (attribute access); demos restored via
    `Module.load()` come back as plain dicts (item access). `getattr` on a dict silently
    returns the default, which would have reported every loaded demonstration as having
    zero-length reasoning -- and hence as trivially brevity-compliant. Both call sites
    that inspect saved programs depend on this."""
    if isinstance(demo, dict):
        return demo.get(name, default) or default
    return getattr(demo, name, default) or default


def demo_payload(demos) -> dict:
    """Size of a demonstration set, in the two units the paper quotes: reasoning
    characters (the register the optimizer imports) and full rendered payload
    (question + options + reasoning, what actually consumes the context)."""
    reasoning = [len(demo_field(d, "reasoning")) for d in demos]
    full = [len(demo_field(d, "question")) + len(demo_field(d, "options"))
            + len(demo_field(d, "reasoning")) for d in demos]
    return {"n_demos": len(demos), "reasoning_chars": reasoning,
            "reasoning_total": sum(reasoning), "payload_total": sum(full),
            "n_compliant": sum(1 for d in demos
                               if is_compliant(demo_field(d, "reasoning")))}


def rescue_letter(raw: str, letters: str = CHOICE_LETTERS) -> str:
    """Extract an answer letter from a raw completion that the adapter failed to parse."""
    if not raw:
        return ""
    m = _adapter_answer_re(letters).search(raw)
    if m:
        return m.group(1).upper()
    result = parse_letter(raw, letters)
    return result if result in letters else ""


class DirectAnswer(dspy.Signature):
    """O'zbek tilidagi test savoliga javob ber."""
    question: str = dspy.InputField()
    options: str = dspy.InputField()
    answer_letter: str = dspy.OutputField(desc="Faqat bitta harf: A, B, C yoki D.")


class DirectSolver(dspy.Module):
    """No-reasoning reference condition (the paper ladder's 'baseline')."""
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(DirectAnswer)

    def forward(self, question, options):
        r = self.predict(question=question, options=options)
        return dspy.Prediction(answer_letter=r.answer_letter, reasoning="")
