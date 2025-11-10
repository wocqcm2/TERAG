"""
Configuration loader for TERAG framework.
Loads and manages configuration from YAML files.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class APIConfig:
    """API configuration."""
    provider: str = "deepinfra"
    deepinfra_api_key: str = ""
    deepinfra_base_url: str = "https://api.deepinfra.com/v1/openai"
    deepinfra_extraction_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    deepinfra_generation_model: str = "meta-llama/Llama-3.3-70B-Instruct"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_extraction_model: str = "gpt-4o-mini"
    openai_generation_model: str = "gpt-4o"
    timeout: int = 260
    max_retries: int = 5


@dataclass
class ExtractionConfig:
    """Concept extraction configuration."""
    text_chunk_size: int = 4096
    chunk_overlap: int = 150
    remove_doc_spaces: bool = True
    batch_size: int = 320
    optimized_batching: bool = True
    optimized_batch_size: int = 5
    max_workers: int = 16
    sleep_interval: float = 0.1
    batch_rest_frequency: int = 50
    max_new_tokens: int = 1024
    temperature: float = 0.1
    normalize_concept_names: bool = True
    filter_low_quality: bool = True
    min_concept_frequency: int = 1
    extraction_mode: str = "passage_entity_document_concept"
    language: str = "en"
    include_abstraction_levels: bool = True
    include_hierarchical_relations: bool = True
    record_usage: bool = True
    save_intermediate_results: bool = True


@dataclass
class GraphConfig:
    """Graph construction configuration."""
    create_passage_nodes: bool = True
    create_concept_nodes: bool = True
    create_has_passage_edges: bool = True
    create_cooccur_edges: bool = True
    max_concepts_per_passage: int = 500
    cooccur_bidirectional: bool = True
    enable_clustering: bool = False
    clustering_method: str = "fast"
    output_format: str = "graphml"
    include_node_attributes: bool = True
    include_edge_attributes: bool = True


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    embedding_model: str = "all-MiniLM-L6-v2"
    normalize_embeddings: bool = True
    embedding_batch_size: int = 64
    topk_retrieval: int = 5
    include_concept_nodes: bool = True
    include_event_nodes: bool = True
    
    # HippoRAG original settings
    original_topk_nodes: int = 10
    original_ppr_alpha: float = 0.85
    original_ppr_max_iter: int = 2000
    original_ppr_tol: float = 1e-7
    
    # HippoRAG enhanced settings
    enhanced_topk_nodes: int = 30
    enhanced_ppr_alpha: float = 0.4
    enhanced_ppr_max_iter: int = 3000
    enhanced_ppr_tol: float = 1e-6
    enhanced_freq_weight_factor: float = 0.5


@dataclass
class BenchmarkConfig:
    """Benchmark evaluation configuration."""
    datasets: Dict[str, str] = field(default_factory=dict)
    num_samples: int = -1
    topk_retrieval: int = 5
    compute_em: bool = True
    compute_f1: bool = True
    compute_recall_at_2: bool = True
    compute_recall_at_5: bool = True
    max_context_length: int = 1500
    max_per_document: int = 300
    generation_max_tokens: int = 2048
    generation_temperature: float = 0.5
    save_detailed_results: bool = True
    save_summary: bool = True
    log_level: str = "INFO"


@dataclass
class GeneralConfig:
    """General configuration."""
    output_dir: str = "output"
    log_dir: str = "logs"
    debug_mode: bool = False
    verbose: bool = True
    random_seed: int = 42


class ConfigLoader:
    """
    Configuration loader for TERAG framework.
    Loads configuration from YAML files and provides typed access.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to configuration YAML file.
                        If None, uses default config/config.yaml
        """
        if config_path is None:
            # Try to find config in standard locations
            possible_paths = [
                Path("config/config.yaml"),
                Path("../config/config.yaml"),
                Path(__file__).parent.parent.parent / "config" / "config.yaml",
            ]
            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break
        
        if config_path is None:
            raise FileNotFoundError(
                "Could not find config.yaml. Please specify config_path."
            )
        
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load raw configuration
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.raw_config: Dict[str, Any] = yaml.safe_load(f)
        
        # Parse into structured config objects
        self.api = self._load_api_config()
        self.extraction = self._load_extraction_config()
        self.graph = self._load_graph_config()
        self.retrieval = self._load_retrieval_config()
        self.benchmark = self._load_benchmark_config()
        self.general = self._load_general_config()
    
    def _load_api_config(self) -> APIConfig:
        """Load API configuration."""
        api_data = self.raw_config.get("api", {})
        provider = api_data.get("provider", "deepinfra")
        
        deepinfra_data = api_data.get("deepinfra", {})
        openai_data = api_data.get("openai", {})
        
        return APIConfig(
            provider=provider,
            deepinfra_api_key=deepinfra_data.get("api_key", ""),
            deepinfra_base_url=deepinfra_data.get("base_url", "https://api.deepinfra.com/v1/openai"),
            deepinfra_extraction_model=deepinfra_data.get("extraction_model", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
            deepinfra_generation_model=deepinfra_data.get("generation_model", "meta-llama/Llama-3.3-70B-Instruct"),
            openai_api_key=openai_data.get("api_key", ""),
            openai_base_url=openai_data.get("base_url", "https://api.openai.com/v1"),
            openai_extraction_model=openai_data.get("extraction_model", "gpt-4o-mini"),
            openai_generation_model=openai_data.get("generation_model", "gpt-4o"),
            timeout=deepinfra_data.get("timeout", 260),
            max_retries=deepinfra_data.get("max_retries", 5),
        )
    
    def _load_extraction_config(self) -> ExtractionConfig:
        """Load extraction configuration."""
        data = self.raw_config.get("extraction", {})
        return ExtractionConfig(
            text_chunk_size=data.get("text_chunk_size", 4096),
            chunk_overlap=data.get("chunk_overlap", 150),
            remove_doc_spaces=data.get("remove_doc_spaces", True),
            batch_size=data.get("batch_size", 320),
            optimized_batching=data.get("optimized_batching", True),
            optimized_batch_size=data.get("optimized_batch_size", 5),
            max_workers=data.get("max_workers", 16),
            sleep_interval=data.get("sleep_interval", 0.1),
            batch_rest_frequency=data.get("batch_rest_frequency", 50),
            max_new_tokens=data.get("max_new_tokens", 1024),
            temperature=data.get("temperature", 0.1),
            normalize_concept_names=data.get("normalize_concept_names", True),
            filter_low_quality=data.get("filter_low_quality", True),
            min_concept_frequency=data.get("min_concept_frequency", 1),
            extraction_mode=data.get("extraction_mode", "passage_entity_document_concept"),
            language=data.get("language", "en"),
            include_abstraction_levels=data.get("include_abstraction_levels", True),
            include_hierarchical_relations=data.get("include_hierarchical_relations", True),
            record_usage=data.get("record_usage", True),
            save_intermediate_results=data.get("save_intermediate_results", True),
        )
    
    def _load_graph_config(self) -> GraphConfig:
        """Load graph configuration."""
        data = self.raw_config.get("graph", {})
        return GraphConfig(
            create_passage_nodes=data.get("create_passage_nodes", True),
            create_concept_nodes=data.get("create_concept_nodes", True),
            create_has_passage_edges=data.get("create_has_passage_edges", True),
            create_cooccur_edges=data.get("create_cooccur_edges", True),
            max_concepts_per_passage=data.get("max_concepts_per_passage", 500),
            cooccur_bidirectional=data.get("cooccur_bidirectional", True),
            enable_clustering=data.get("enable_clustering", False),
            clustering_method=data.get("clustering_method", "fast"),
            output_format=data.get("output_format", "graphml"),
            include_node_attributes=data.get("include_node_attributes", True),
            include_edge_attributes=data.get("include_edge_attributes", True),
        )
    
    def _load_retrieval_config(self) -> RetrievalConfig:
        """Load retrieval configuration."""
        data = self.raw_config.get("retrieval", {})
        original = data.get("hipporag_original", {})
        enhanced = data.get("hipporag_enhanced", {})
        
        return RetrievalConfig(
            embedding_model=data.get("embedding_model", "all-MiniLM-L6-v2"),
            normalize_embeddings=data.get("normalize_embeddings", True),
            embedding_batch_size=data.get("embedding_batch_size", 64),
            topk_retrieval=data.get("topk_retrieval", 5),
            include_concept_nodes=data.get("include_concept_nodes", True),
            include_event_nodes=data.get("include_event_nodes", True),
            original_topk_nodes=original.get("topk_nodes", 10),
            original_ppr_alpha=original.get("ppr_alpha", 0.85),
            original_ppr_max_iter=original.get("ppr_max_iter", 2000),
            original_ppr_tol=original.get("ppr_tol", 1e-7),
            enhanced_topk_nodes=enhanced.get("topk_nodes", 30),
            enhanced_ppr_alpha=enhanced.get("ppr_alpha", 0.4),
            enhanced_ppr_max_iter=enhanced.get("ppr_max_iter", 3000),
            enhanced_ppr_tol=enhanced.get("ppr_tol", 1e-6),
            enhanced_freq_weight_factor=enhanced.get("freq_weight_factor", 0.5),
        )
    
    def _load_benchmark_config(self) -> BenchmarkConfig:
        """Load benchmark configuration."""
        data = self.raw_config.get("benchmark", {})
        return BenchmarkConfig(
            datasets=data.get("datasets", {}),
            num_samples=data.get("num_samples", -1),
            topk_retrieval=data.get("topk_retrieval", 5),
            compute_em=data.get("compute_em", True),
            compute_f1=data.get("compute_f1", True),
            compute_recall_at_2=data.get("compute_recall_at_2", True),
            compute_recall_at_5=data.get("compute_recall_at_5", True),
            max_context_length=data.get("max_context_length", 1500),
            max_per_document=data.get("max_per_document", 300),
            generation_max_tokens=data.get("generation_max_tokens", 2048),
            generation_temperature=data.get("generation_temperature", 0.5),
            save_detailed_results=data.get("save_detailed_results", True),
            save_summary=data.get("save_summary", True),
            log_level=data.get("log_level", "INFO"),
        )
    
    def _load_general_config(self) -> GeneralConfig:
        """Load general configuration."""
        data = self.raw_config.get("general", {})
        return GeneralConfig(
            output_dir=data.get("output_dir", "output"),
            log_dir=data.get("log_dir", "logs"),
            debug_mode=data.get("debug_mode", False),
            verbose=data.get("verbose", True),
            random_seed=data.get("random_seed", 42),
        )
    
    def get_extraction_model(self) -> str:
        """Get the extraction model name based on provider."""
        if self.api.provider == "deepinfra":
            return self.api.deepinfra_extraction_model
        elif self.api.provider == "openai":
            return self.api.openai_extraction_model
        else:
            raise ValueError(f"Unknown API provider: {self.api.provider}")
    
    def get_generation_model(self) -> str:
        """Get the generation model name based on provider."""
        if self.api.provider == "deepinfra":
            return self.api.deepinfra_generation_model
        elif self.api.provider == "openai":
            return self.api.openai_generation_model
        else:
            raise ValueError(f"Unknown API provider: {self.api.provider}")
    
    def get_api_key(self) -> str:
        """Get the API key based on provider."""
        if self.api.provider == "deepinfra":
            return self.api.deepinfra_api_key
        elif self.api.provider == "openai":
            return self.api.openai_api_key
        else:
            raise ValueError(f"Unknown API provider: {self.api.provider}")
    
    def get_base_url(self) -> str:
        """Get the base URL based on provider."""
        if self.api.provider == "deepinfra":
            return self.api.deepinfra_base_url
        elif self.api.provider == "openai":
            return self.api.openai_base_url
        else:
            raise ValueError(f"Unknown API provider: {self.api.provider}")

