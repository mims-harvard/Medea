"""
Medea: Multi-Agent System for Single-Cell Analysis

A modular agent system for autonomous research planning, in-silico experimentation,
and literature-based reasoning in therapeutic discovery.
"""

__version__ = "1.0.0"
__author__ = "Medea Team"

# Core workflow functions
from .core import medea, experiment_analysis, literature_reasoning

# Agent classes for custom workflows
from .agents import (
    ResearchPlanning,
    Analysis,
    LiteratureReasoning,
    AgentLLM,
    LLMConfig
)

# Action classes for custom agent configurations
from .agents.research_planning import (
    ResearchPlanDraft,
    ContextVerification,
    IntegrityVerification
)

from .agents.experiment_analysis import (
    CodeGenerator,
    AnalysisExecution,
    CodeDebug,
    CodeQulityChecker
)

from .agents.literature_reasoning import (
    LiteratureSearch,
    PaperJudge,
    OpenScholarReasoning
)

# Utility classes
from .agents.utils import Proposal, CodeSnippet

# Panel discussion
from .agents.discussion import multi_round_discussion

__all__ = [
    # Core functions
    'medea',
    'experiment_analysis',
    'literature_reasoning',
    
    # Agent classes
    'ResearchPlanning',
    'Analysis',
    'LiteratureReasoning',
    'AgentLLM',
    'LLMConfig',
    
    # Research planning actions
    'ResearchPlanDraft',
    'ContextVerification',
    'IntegrityVerification',
    
    # Analysis actions
    'CodeGenerator',
    'AnalysisExecution',
    'CodeDebug',
    'CodeQulityChecker',
    
    # Literature reasoning actions
    'LiteratureSearch',
    'PaperJudge',
    'OpenScholarReasoning',
    
    # Utilities
    'Proposal',
    'CodeSnippet',
    'multi_round_discussion',
]

