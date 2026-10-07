"""
Verification Agent: check the master's answer before the user sees it.

Deterministic checks (always on, zero tokens, ~1 ms):
  citation precision   cited labels that point to real evidence
  citation coverage    factual paragraphs that carry at least one citation
  numerical grounding  every number in the answer appears in the evidence,
                       the question, or a calculator result (mA<->A etc. allowed)
  equation grounding   symbols in display equations appear in the evidence
  completeness         the answer actually answers (not empty / not cut off)

Optional LLM check (VERIFY_WITH_LLM=true, uses the "verifier" model).

These are application-level checks, not proofs of correctness.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from config.settings import get_logger
from llm.model_router import ModelRouter
from llm.providers import LLMError
from rag.agent import NOT_FOUND

log = get_logger("verifier")

CITE = re.compile(r"\[REF-(\d+)[^\]]*\]")
_NUM = re.compile(r"(?<![A-Za-z_\d.])-?\d+(?:\.\d+)?(?![\d])")
_IDS = re.compile(r"\b(?:EQ|TBL|FIG)_\d+\b|\bREF-\d+\b|\bp\d+\b")
_DISPLAY_EQ = re.compile(r"\$\$(.+?)\$\$", re.S)
_EQ_SYMBOL = re.compile(r"\\?[A-Za-z]+(?:_\{?[A-Za-z0-9\\ ]+\}?)")


@dataclass
class VerificationReport:
    grounded: bool = True
    verdict: str = "verified"  # verified | partial | unverified | n/a
    citation_precision: float = 1.0  # citations that point to real evidence
    citation_coverage: float = 1.0  # factual paragraphs with a citation
    citation_accuracy: float = 1.0  # cited values found in THEIR OWN cited source
    citation_support: float = 1.0  # cited sentences the cross-encoder judges supported
    numerical_score: float = 1.0
    equation_score: float = 1.0
    completeness: float = 1.0
    unsupported_claims: List[str] = field(default_factory=list)
    miscited: List[str] = field(default_factory=list)
    weak_support: List[str] = field(default_factory=list)
    invalid_citations: List[str] = field(default_factory=list)
    needs_regeneration: bool = False
    llm_checked: bool = False
    notes: List[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


_UNIT = re.compile(r"^\s*(k|M|m|µ|u|n|p)?(A|V|W|Hz|Ω|ohm|F|s|°C)\b")
_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s")
SupportFn = Callable[[str, List[str]], List[float]]


def _numbers_with_units(text: str) -> List[Tuple[float, bool]]:
    """(value, has_prefixed_or_base_unit) for each number, ignoring citations/ids/list markers."""
    clean = _IDS.sub(" ", CITE.sub(" ", _LIST_MARKER.sub(" ", text)))
    out = []
    for m in _NUM.finditer(clean):
        try:
            value = float(m.group(0))
        except ValueError:
            continue
        out.append((value, bool(_UNIT.match(clean[m.end():]))))
    return out


def _numbers(text: str) -> List[float]:
    return [v for v, _ in _numbers_with_units(text)]


def _found(value: float, has_unit: bool, pool: Sequence[float]) -> bool:
    scales = (1.0, 1000.0, 0.001) if has_unit else (1.0,)  # 500 mA <-> 0.5 A only when a unit is given
    return any(abs(value - p * sc) <= max(1e-9, 0.005 * abs(value)) for p in pool for sc in scales)


def _norm_symbol(sym: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", sym.replace("\\", "")).upper()


def _paragraphs(answer: str) -> List[str]:
    parts = re.split(r"\n\s*\n|\n(?=\s*[-*•]\s)|\n(?=\s*\d+\.\s)", answer)
    return [p.strip() for p in parts if p.strip()]


def _sentences(par: str) -> List[str]:
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z*$])|\n", par) if x.strip()]


def _is_factual(par: str) -> bool:
    plain = re.sub(r"\*\*[^*]+:\*\*", "", par).strip()
    if not plain or plain.startswith("$$") or NOT_FOUND in plain.upper():
        return False
    return bool(re.search(r"\d", plain)) or len(plain.split()) >= 8


def check(answer: str, ref_texts: Dict[int, str], observations: Sequence[str], question: str,
          invalid_citations: Sequence[str], support_fn: Optional[SupportFn] = None,
          support_threshold: float = 0.0) -> VerificationReport:
    """Claim-level verification. `ref_texts` maps REF number -> evidence text."""
    t0 = time.perf_counter()
    rep = VerificationReport(invalid_citations=list(invalid_citations))
    if not answer.strip() or NOT_FOUND in answer.upper()[:60]:
        rep.completeness = 0.0 if not answer.strip() else 1.0
        rep.verdict = "n/a"
        rep.latency_ms = (time.perf_counter() - t0) * 1000
        return rep

    labels = CITE.findall(answer)
    total_cites = len(labels) + len(invalid_citations)
    rep.citation_precision = round(len(labels) / total_cites, 3) if total_cites else 0.0
    factual = [p for p in _paragraphs(answer) if _is_factual(p)]
    if factual:
        rep.citation_coverage = round(sum(1 for p in factual if CITE.search(p)) / len(factual), 3)

    ref_nums = {n: _numbers(t) for n, t in ref_texts.items()}
    all_nums = [v for vals in ref_nums.values() for v in vals]
    calc_nums = _numbers("\n".join(observations))  # calculator / tool results
    free_nums = calc_nums + _numbers(question)  # + values given in the question
    body = _DISPLAY_EQ.sub(" ", answer)
    n_total = n_ok = cited_total = cited_ok = 0
    support_pairs: List[Tuple[str, List[int]]] = []
    for par in _paragraphs(body):
        par_refs = [int(x) for x in CITE.findall(par)]
        for sent in _sentences(par):
            refs = [int(x) for x in CITE.findall(sent)] or par_refs
            refs = [r for r in refs if r in ref_texts]
            sent_nums = _numbers_with_units(sent)
            from_calculator = bool(sent_nums) and all(_found(v, u, calc_nums) for v, u in sent_nums)
            if refs and not from_calculator and re.search(r"[A-Za-z]{3}", CITE.sub("", sent)):
                # (calculated results are supported by the calculation itself, not by the text)
                support_pairs.append((CITE.sub("", sent).strip(), refs))
            own = [v for r in refs for v in ref_nums.get(r, [])]
            for value, has_unit in sent_nums:
                n_total += 1
                if refs:
                    cited_total += 1
                if refs and _found(value, has_unit, own):
                    n_ok += 1
                    cited_ok += 1
                elif _found(value, has_unit, free_nums):
                    n_ok += 1
                    cited_ok += 1 if refs else 0  # computed / given values are fine under any citation
                elif _found(value, has_unit, all_nums):
                    n_ok += 1  # the value exists, but in a different source
                    if refs:
                        holder = next(r for r, vals in ref_nums.items() if _found(value, has_unit, vals))
                        rep.miscited.append(f"{value:g} is in REF-{holder}, not REF-{', REF-'.join(map(str, refs))}")
                else:
                    rep.unsupported_claims.append(CITE.sub("", sent).strip()[:160])
    if n_total:
        rep.numerical_score = round(n_ok / n_total, 3)
    if cited_total:
        rep.citation_accuracy = round(cited_ok / cited_total, 3)
    rep.unsupported_claims = list(dict.fromkeys(rep.unsupported_claims))[:5]
    rep.miscited = list(dict.fromkeys(rep.miscited))[:5]

    # Semantic support: does the cited excerpt actually back the sentence? (local cross-encoder)
    if support_fn and support_pairs:
        ok = 0
        for sent, refs in support_pairs[:12]:
            try:
                best = max(support_fn(sent, [ref_texts[r] for r in refs]))
            except Exception:
                best = support_threshold  # scoring unavailable: don't penalise
            if best >= support_threshold:
                ok += 1
            else:
                rep.weak_support.append(sent[:140])
        rep.citation_support = round(ok / min(len(support_pairs), 12), 3)

    corpus_norm = _norm_symbol("\n".join([*ref_texts.values(), *observations, question]))
    symbols = [_norm_symbol(x) for eq in _DISPLAY_EQ.findall(answer) for x in _EQ_SYMBOL.findall(eq)]
    symbols = [x for x in dict.fromkeys(symbols) if len(x) >= 2 and x not in {"TIMES", "CDOT", "FRAC", "LEFT", "RIGHT"}]
    if symbols:
        rep.equation_score = round(sum(1 for x in symbols if x in corpus_norm) / len(symbols), 3)

    if len(answer.strip()) < 20 or answer.rstrip().endswith((",", " and", " the", "(")):
        rep.completeness = 0.5
    rep.grounded = (not invalid_citations and not rep.miscited and rep.numerical_score >= 0.9
                    and rep.equation_score >= 0.6 and rep.citation_precision > 0)
    rep.needs_regeneration = not rep.grounded
    rep.verdict = ("unverified" if not rep.grounded else
                   "partial" if (rep.citation_support < 0.75 or rep.citation_coverage < 0.75) else "verified")
    rep.latency_ms = (time.perf_counter() - t0) * 1000
    return rep


def llm_check(rep: VerificationReport, answer: str, evidence_text: str, router: ModelRouter) -> VerificationReport:
    """Optional second opinion from a small model (VERIFY_WITH_LLM=true)."""
    t0 = time.perf_counter()
    prompt = ("You are a strict verifier. Compare the ANSWER with the EVIDENCE. Reply with JSON only: "
              '{"grounded": true|false, "unsupported_claims": ["..."]}.\n\n'
              f"EVIDENCE:\n{evidence_text[:4000]}\n\nANSWER:\n{answer[:2500]}")
    try:
        raw = router.chat("verifier", [{"role": "user", "content": prompt}], max_tokens=400).result.content
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        rep.llm_checked = True
        if data.get("grounded") is False:
            rep.grounded = False
            rep.needs_regeneration = True
            rep.unsupported_claims += [str(c)[:160] for c in data.get("unsupported_claims", [])][:5]
    except (LLMError, AttributeError, ValueError, TypeError) as exc:
        rep.notes.append(f"LLM verification skipped ({type(exc).__name__}).")
    rep.latency_ms += (time.perf_counter() - t0) * 1000
    return rep


def feedback_text(rep: VerificationReport) -> str:
    parts = []
    if rep.invalid_citations:
        parts.append(f"These citations do not exist: {', '.join(rep.invalid_citations)}.")
    if rep.miscited:
        parts.append("These values are cited to the wrong source; fix the citations: " + "; ".join(rep.miscited))
    if rep.unsupported_claims:
        parts.append("These statements contain values not found in the evidence: "
                     + " | ".join(rep.unsupported_claims[:4]))
    if rep.equation_score < 0.6:
        parts.append("Some equation symbols do not appear in the evidence; use only equations from the sources.")
    return " ".join(parts) or "Make sure every fact is supported and cited."


def canonicalize(answer: str, by_number) -> Tuple[str, list, List[str]]:
    """Rewrite citations to their true labels. Returns (answer, cited refs, invalid labels)."""
    cited, bad = [], []

    def _sub(m: "re.Match[str]") -> str:
        ref = by_number(int(m.group(1)))
        if ref is None:
            bad.append(m.group(0))
            return m.group(0)
        if ref not in cited:
            cited.append(ref)
        return ref.label

    return CITE.sub(_sub, answer), cited, list(dict.fromkeys(bad))

