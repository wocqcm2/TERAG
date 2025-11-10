"""
TERAG: Text-based Entity Relation Augmented Generation

A professional framework for knowledge graph construction and 
retrieval-augmented generation (RAG) on multi-hop QA datasets.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from terag.extraction import ConceptExtractor
from terag.graph import GraphBuilder
from terag.retrieval import HippoRAGRetriever, HippoRAGEnhancedRetriever
from terag.benchmark import RAGEvaluator

__all__ = [
    "ConceptExtractor",
    "GraphBuilder",
    "HippoRAGRetriever",
    "HippoRAGEnhancedRetriever",
    "RAGEvaluator",
]

