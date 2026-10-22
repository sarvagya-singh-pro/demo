"""Canonical trade taxonomy and risk-tier resolution.

WHY THIS MODULE EXISTS
----------------------
The risk tier used to be derived by naive substring matching on the free-text
trade description (``"tow" in "town plumbing"`` -> high risk). That produced
two distinct defects:

1. **False matches.** ``"town plumbing"`` matched the high-risk keyword
   ``"tow"`` and was classed as *high* risk.
2. **Same trade, different tier.** The same underlying trade, worded
   differently, landed in different tiers::

       handyman   ['low', 'medium', 'medium', 'medium']
       gardener   ['low', 'low', 'medium', 'medium']
       demolition ['high', 'medium', 'high', 'high']
       waste      ['high', 'high', 'medium', 'high']

   That is not a realistic source of risk difference - it is a wording
   artefact, and it was manufacturing decision flips in the consistency test.

THE FIX
-------
Risk tier is a property of the *trade*, not of the sentence describing it.
So the tier is resolved in two steps:

1. **Identify the trade** from the text using word-boundary matching over a
   canonical vocabulary of trades and their synonym phrases. All wordings of
   one trade resolve to the SAME trade key, hence the same tier.
2. **Fall back** to word-boundary tier keywords only for trades that are not
   in the canonical vocabulary.

All matching is done on whole words (or whole phrases), never on raw
substrings, so ``"tow"`` no longer matches ``"town"``.

Both ``engine.py`` and ``ml_engine.py`` import from here so the two engines
can never drift apart.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Canonical trades and their risk tier
# ---------------------------------------------------------------------------

# Trade key -> risk tier. These tiers are the business judgement of the
# demo's data owner: trades whose work is mostly indoor/low-consequence are
# low risk, trades whose work is weather-exposed and revenue-critical are
# higher risk.
TRADE_TIERS: Dict[str, str] = {
    # Low risk: low-consequence, largely indoor, replaceable-equipment work.
    "electrician": "low",
    "plumber": "low",
    "carpenter": "low",
    "painter": "low",
    "handyman": "low",
    "gardener": "low",
    "cleaner": "low",
    # Medium risk: project-based, weather-exposed, cash-flow sensitive.
    "builder": "medium",
    "roofer": "medium",
    "plasterer": "medium",
    "glazier": "medium",
    "flooring": "medium",
    "fencing": "medium",
    "arborist": "medium",
    "pest_control": "medium",
    # High risk: hazardous, regulated, or heavy-goods operations.
    "demolition": "high",
    "waste": "high",
    "asbestos": "high",
    "scaffolding": "high",
    "excavation": "high",
    "crane": "high",
    "scrap": "high",
}


# Synonym phrases -> canonical trade key. Every phrase here is matched on
# WHOLE WORDS. Matching prefers the longest phrase so that "waste disposal"
# wins over a bare "disposal".
TRADE_SYNONYMS: Dict[str, List[str]] = {
    "electrician": [
        "electrician", "electrical contractor", "electrical installation",
        "domestic electrician",
    ],
    "plumber": [
        "plumber", "plumbing services", "plumbing and heating engineer",
        "plumbing and heating", "domestic plumber", "plumbing engineer",
        "plumbing & heating engineer", "plumbing & heating",
    ],
    "carpenter": ["carpenter", "carpentry", "joiner", "joinery"],
    "painter": [
        "painter", "painting and decorating", "painting & decorating",
        "decorator", "decorating", "tiler", "tiling",
    ],
    "handyman": [
        "handyman", "odd job person", "odd-job person", "odd job",
        "property maintenance", "home repairs", "home repair",
        "general handyman", "general repairer",
    ],
    "gardener": [
        "gardener", "landscape gardener", "garden maintenance",
        "groundskeeping", "groundskeeper", "landscaping",
    ],
    "cleaner": [
        "cleaner", "cleaning contractor", "commercial cleaning",
        "office cleaning services", "cleaning services",
    ],
    "builder": [
        "builder", "general building contractor", "residential builder",
        "building works", "building contractor", "building work",
    ],
    "roofer": ["roofer", "roofing", "roofing contractor", "roofing services"],
    "plasterer": ["plasterer", "plastering", "dry lining", "drylining"],
    "glazier": ["glazier", "glazing", "glazing contractor"],
    "flooring": [
        "flooring", "floor fitter", "floor installation",
        "flooring contractor", "flooring specialist",
    ],
    "fencing": [
        "fencing", "fence contractor", "fence installation",
        "fencing contractor", "fencing and gates",
    ],
    "arborist": [
        "arborist", "tree surgeon", "tree surgery", "tree care",
    ],
    "pest_control": [
        "pest control", "pest controller", "pest control contractor",
        "pest management",
    ],
    "demolition": [
        "demolition contractor", "demolition and wrecking",
        "demolition & wrecking", "structural demolition", "site clearance",
        "demolition", "wrecking contractor",
    ],
    "waste": [
        "waste collection", "waste management", "waste disposal services",
        "rubbish removal", "waste disposal", "waste", "house clearance",
        "skip hire",
    ],
    "asbestos": ["asbestos removal", "asbestos"],
    "scaffolding": ["scaffolding contractor", "scaffolding", "scaffold hire"],
    "excavation": [
        "excavation", "groundworks", "groundworks contractor",
        "site excavation",
    ],
    "crane": ["crane hire", "crane operator", "crane", "crane hire services"],
    "scrap": [
        "scrap metal", "scrap metal merchant", "scrap", "scrap metal dealer",
    ],
}

# Tier keywords used ONLY when the trade could not be identified from the
# canonical vocabulary. These are matched on whole words too.
TIER_KEYWORDS: Dict[str, List[str]] = {
    "high": [
        "demolition", "asbestos", "waste", "skip", "hazardous", "crane",
        "scaffolding", "excavation", "scrap", "tow",
    ],
    "medium": [
        "builder", "building", "roofer", "roofing", "plasterer", "glazier",
        "flooring", "fence", "fencing", "tree", "arborist", "pest", "removal",
    ],
    "low": [
        "electrician", "electrical", "plumber", "plumbing", "carpenter",
        "carpentry", "joiner", "painter", "decorator", "tiler", "handyman",
        "gardener", "landscape", "cleaner", "cleaning",
    ],
}

# Tier used when nothing matches at all.
DEFAULT_TIER = "medium"



# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def _word_pattern(phrase: str) -> re.Pattern:
    """Compile a whole-word / whole-phrase matcher for ``phrase``.

    Word boundaries are escaped so that punctuation in a phrase (e.g. the
    ampersand in "plumbing & heating engineer") does not break matching.

    The FINAL word accepts an optional plural "s" so that "We are plumbers
    based in London" resolves to the same trade as "plumber". Without this,
    a business describing itself in the plural was silently falling through
    to the default (medium) tier - a wording artefact masquerading as risk.
    """
    parts = [re.escape(p) for p in phrase.split()]
    if not parts:
        return re.compile(r"(?!x)x")  # never matches
    body = r"\s+".join(parts)
    # Optional plural on the last word only.
    return re.compile(r"(?<!\w)" + body + r"s?(?!\w)")


# Pre-compile every pattern once at import time.
_SYNONYM_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (_word_pattern(phrase), key)
    for key, phrases in TRADE_SYNONYMS.items()
    for phrase in sorted(phrases, key=len, reverse=True)
]

_TIER_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (_word_pattern(kw), tier)
    for tier, kws in TIER_KEYWORDS.items()
    for kw in sorted(kws, key=len, reverse=True)
]


def identify_trade(trade: str) -> str:
    """Return the canonical trade key for a free-text trade description.

    Matching is whole-word, and the longest matching phrase wins. Returns an
    empty string when the trade cannot be identified.
    """
    t = (trade or "").lower().strip()
    if not t:
        return ""
    best_key = ""
    best_len = 0
    for pattern, key in _SYNONYM_PATTERNS:
        m = pattern.search(t)
        if m:
            # Prefer the longest match; ties broken by canonical key order.
            if len(m.group(0)) > best_len:
                best_key, best_len = key, len(m.group(0))
    return best_key


def normalise_trade_category(trade: str) -> str:
    """Map a free-text trade description to a risk tier.

    The tier depends on the *trade*, not on the wording, so all paraphrases
    of one trade resolve to the same tier. Unknown trades fall back to
    whole-word tier keywords and then to ``DEFAULT_TIER``.
    """
    key = identify_trade(trade)
    if key:
        return TRADE_TIERS[key]

    t = (trade or "").lower().strip()
    for pattern, tier in _TIER_PATTERNS:
        if pattern.search(t):
            return tier
    return DEFAULT_TIER
