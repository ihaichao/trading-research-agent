"""Agent 层：把工具拿到的证据，变成一份带引用的报告。"""

from tra.agent.evidence import EvidencePack, build_evidence
from tra.agent.llm import LLMCallable, openai_llm
from tra.agent.pipeline import research
from tra.agent.synthesize import draft_sections

__all__ = [
    "EvidencePack",
    "LLMCallable",
    "build_evidence",
    "draft_sections",
    "openai_llm",
    "research",
]
