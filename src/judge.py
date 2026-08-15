"""LLM-as-judge scoring for open-ended financial answers."""


def build_judge_prompt(question: str, reference: str, answer: str) -> str:
    raise NotImplementedError


def parse_score(text: str) -> dict[str, float]:
    raise NotImplementedError
