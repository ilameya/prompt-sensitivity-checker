import re


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_yes_no(text: str) -> str:
    """
    Robustly extract yes/no from model output.
    Returns: 'yes' / 'no' / 'unknown'
    """
    t = normalize_text(text)
    if re.search(r"\byes\b", t):
        return "yes"
    if re.search(r"\bno\b", t):
        return "no"
    return "unknown"


def is_correct_boolq(pred_text: str, gold: str) -> int:
    pred = extract_yes_no(pred_text)
    return int(pred == gold)