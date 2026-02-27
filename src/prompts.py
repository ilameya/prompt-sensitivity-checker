from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class PromptVariant:
    name: str
    template: str

    def format(self, question: str) -> str:
        return self.template.format(question=question)


VARIANTS: List[PromptVariant] = [
    PromptVariant(
        name="direct",
        template=(
            "Answer ONLY with 'yes' or 'no'. No explanation.\n"
            "Question: {question}\n"
            "Answer:"
        ),
    ),
    PromptVariant(
        name="role_expert",
        template=(
            "You are a careful expert assistant.\n"
            "Answer ONLY with 'yes' or 'no'. No explanation.\n"
            "Question: {question}\n"
            "Answer:"
        ),
    ),
    PromptVariant(
        name="constraints",
        template=(
            "Rules:\n"
            "1) Output exactly one token: yes OR no\n"
            "2) No punctuation\n"
            "3) No extra words\n"
            "Question: {question}\n"
            "Answer:"
        ),
    ),
    PromptVariant(
        name="step_by_step",
        template=(
            "Think privately, then answer ONLY with 'yes' or 'no'. No explanation.\n"
            "Question: {question}\n"
            "Answer:"
        ),
    ),
    PromptVariant(
        name="minimal",
        template=(
            "yes/no only\n"
            "{question}\n"
            "Answer:"
        ),
    ),
]


def get_variants() -> Dict[str, PromptVariant]:
    return {v.name: v for v in VARIANTS}