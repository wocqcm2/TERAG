#!/usr/bin/env python3
"""
TERAG End-to-End Pipeline
Complete workflow: Concept Extraction -> Graph Building -> Benchmark Evaluation
"""

import argparse
import logging
import sys
import json
import torch
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from terag.utils.config_loader import ConfigLoader
from terag.utils.llm_client import LLMClient
from terag.extraction import ConceptExtractor
from terag.graph import GraphBuilder
from terag.retrieval import HippoRAGRetriever, HippoRAGEnhancedRetriever
from terag.benchmark import RAGEvaluator

import pandas as pd
import networkx as nx
from sentence_transformers import SentenceTransformer


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


class TERAGPipeline:
    """
    Complete TERAG pipeline executor.
    Handles extraction, graph building, and benchmark evaluation.
    """
    
    def __init__(self, config: ConfigLoader, output_dir: str):
        """
        Initialize pipeline.
        
        Args:
            config: Configuration loader
            output_dir: Base output directory
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Track intermediate results
        self.concept_file = None
        self.concepts_csv = None
        self.passages_csv = None
        self.graph_file = None
    
    def step1_concept_extraction(
        self,
        dataset_name: str,
        max_samples: int = -1,
        skip_if_exists: bool = False
    ) -> tuple:
        """
        Step 1: Extract concepts from benchmark dataset.
        
        Args:
            dataset_name: Name of dataset (hotpotqa, 2wikimultihopqa, musique)
            max_samples: Maximum samples to process (-1 = all)
            skip_if_exists: Skip if output files already exist
            
        Returns:
            Tuple of (concepts_csv_path, passages_csv_path)
        """
        self.logger.info("=" * 80)
        self.logger.info("STEP 1: Concept Extraction")
        self.logger.info("=" * 80)
        
        extraction_dir = self.output_dir / "extraction"
        
        # Check if already exists
        if skip_if_exists:
            existing_concepts = list((extraction_dir / "concept_csv").glob("concepts_*.csv"))
            existing_passages = list((extraction_dir / "concept_csv").glob("passages_*.csv"))
            
            if existing_concepts and existing_passages:
                self.concepts_csv = str(existing_concepts[-1])
                self.passages_csv = str(existing_passages[-1])
                self.logger.info(f"Skipping extraction - using existing files:")
                self.logger.info(f"  Concepts: {self.concepts_csv}")
                self.logger.info(f"  Passages: {self.passages_csv}")
                return self.concepts_csv, self.passages_csv
        
        # Create LLM client for extraction
        self.logger.info("Initializing LLM client for extraction...")
        extraction_llm = LLMClient(
            api_key=self.config.get_api_key(),
            base_url=self.config.get_base_url(),
            model_name=self.config.get_extraction_model(),
            timeout=self.config.api.timeout,
            max_retries=self.config.api.max_retries
        )
        
        # Create concept extractor
        extractor = ConceptExtractor(
            config=self.config,
            llm_client=extraction_llm,
            output_dir=str(extraction_dir)
        )
        
        # Get data path from config
        dataset_file = self.config.benchmark.datasets.get(dataset_name)
        if dataset_file:
            data_path = str(Path(dataset_file).parent)
        else:
            data_path = "dataset"
        
        # Run extraction
        self.concept_file = extractor.run_extraction(
            dataset_name=dataset_name,
            data_path=data_path,
            max_samples=max_samples
        )
        
        # Create CSV files
        self.logger.info("Creating CSV files...")
        self.concepts_csv, self.passages_csv = extractor.create_csv_files(self.concept_file)
        
        # Aggregate usage statistics
        if self.config.extraction.record_usage:
            self.logger.info("Aggregating token usage...")
            usage_file, usage_summary = extractor.aggregate_usage_stats(self.concept_file)
            self.logger.info(f"Total tokens used: {int(usage_summary['totals']['total_tokens'])}")
        
        self.logger.info("Step 1 completed successfully!")
        self.logger.info(f"Concepts CSV: {self.concepts_csv}")
        self.logger.info(f"Passages CSV: {self.passages_csv}")
        
        return self.concepts_csv, self.passages_csv
    
    def step2_graph_building(self, skip_if_exists: bool = False) -> str:
        """
        Step 2: Build knowledge graph from extracted concepts.
        
        Args:
            skip_if_exists: Skip if graph file already exists
            
        Returns:
            Path to generated GraphML file
        """
        self.logger.info("=" * 80)
        self.logger.info("STEP 2: Graph Construction")
        self.logger.info("=" * 80)
        
        if not self.concepts_csv or not self.passages_csv:
            raise ValueError("Concepts and passages CSV files not available. Run step1 first.")
        
        graph_dir = self.output_dir / "graph"
        graph_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.graph_file = str(graph_dir / f"knowledge_graph_{timestamp}.graphml")
        
        # Check if already exists
        if skip_if_exists and Path(self.graph_file).exists():
            self.logger.info(f"Skipping graph building - using existing file: {self.graph_file}")
            return self.graph_file
        
        # Create graph builder
        builder = GraphBuilder(self.config)
        
        # Build graph
        builder.build_graph(
            concepts_file=self.concepts_csv,
            passages_file=self.passages_csv
        )
        
        # Export to GraphML
        builder.export_to_graphml(self.graph_file)
        
        # Print and save statistics
        builder.print_statistics(self.graph_file)
        
        # Display token usage statistics from extraction phase
        self._display_token_usage()
        
        self.logger.info("Step 2 completed successfully!")
        self.logger.info(f"Graph file: {self.graph_file}")
        self.logger.info(f"Statistics: {Path(self.graph_file).with_suffix('.stats.json')}")
        
        return self.graph_file
    
    def _display_token_usage(self):
        """Display token usage statistics from extraction phase."""
        try:
            # Find the latest usage file
            usage_dir = self.output_dir / "extraction" / "usage"
            if not usage_dir.exists():
                return
            
            usage_files = list(usage_dir.glob("token_usage_*.json"))
            if not usage_files:
                return
            
            # Use the latest file
            latest_usage_file = sorted(usage_files)[-1]
            
            with open(latest_usage_file, 'r', encoding='utf-8') as f:
                usage_data = json.load(f)
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("TOKEN USAGE SUMMARY (Extraction Phase)")
            self.logger.info("=" * 80)
            self.logger.info(f"Model: {usage_data.get('model', 'N/A')}")
            self.logger.info(f"Total API Calls: {usage_data.get('num_calls', 0):,}")
            
            totals = usage_data.get('totals', {})
            self.logger.info(f"\nTotal Tokens:")
            self.logger.info(f"  Prompt Tokens:     {int(totals.get('prompt_tokens', 0)):,}")
            self.logger.info(f"  Completion Tokens: {int(totals.get('completion_tokens', 0)):,}")
            self.logger.info(f"  Total Tokens:      {int(totals.get('total_tokens', 0)):,}")
            self.logger.info(f"  Total Time:        {totals.get('time', 0):.2f}s ({totals.get('time', 0)/60:.2f} min)")
            
            avg = usage_data.get('avg_per_call', {})
            self.logger.info(f"\nAverage Per Call:")
            self.logger.info(f"  Prompt Tokens:     {avg.get('prompt_tokens', 0):.1f}")
            self.logger.info(f"  Completion Tokens: {avg.get('completion_tokens', 0):.1f}")
            self.logger.info(f"  Total Tokens:      {avg.get('total_tokens', 0):.1f}")
            self.logger.info(f"  Time:              {avg.get('time', 0):.2f}s")
            self.logger.info("=" * 80 + "\n")
            
        except Exception as e:
            self.logger.warning(f"Could not load token usage statistics: {e}")
    
    def step3_benchmark_evaluation(
        self,
        dataset_name: str,
        num_samples: int = None,
        retrievers: list = None
    ) -> dict:
        """
        Step 3: Evaluate RAG performance on benchmark.
        
        Args:
            dataset_name: Name of dataset
            num_samples: Number of samples to evaluate (None = use config)
            retrievers: List of retrievers to test ['original', 'enhanced', 'both'] (None = 'both')
            
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info("=" * 80)
        self.logger.info("STEP 3: Benchmark Evaluation")
        self.logger.info("=" * 80)
        
        if not self.graph_file or not self.passages_csv:
            raise ValueError("Graph and passages files not available. Run previous steps first.")
        
        if retrievers is None:
            retrievers = ['both']
        
        # Override num_samples if specified
        if num_samples is not None:
            self.config.benchmark.num_samples = num_samples
        
        # Load graph and passages
        self.logger.info("Loading knowledge graph...")
        kg_graph = nx.read_graphml(self.graph_file)
        self.logger.info(f"Graph loaded: {len(kg_graph.nodes)} nodes, {len(kg_graph.edges)} edges")
        
        self.logger.info("Loading passages...")
        passages_df = pd.read_csv(self.passages_csv)
        self.logger.info(f"Loaded {len(passages_df)} passages")
        
        # Create text dictionary
        text_dict = {}
        for node_id, node_data in kg_graph.nodes(data=True):
            if node_data.get('type') == 'passage' and 'text' in node_data:
                text_dict[node_id] = node_data['text']
        
        self.logger.info(f"Text dictionary: {len(text_dict)} entries")
        
        # Load embedding model
        self.logger.info(f"Loading embedding model: {self.config.retrieval.embedding_model}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"Using device: {device}")
        
        sentence_model = SentenceTransformer(
            self.config.retrieval.embedding_model,
            device=device
        )
        
        # Create embeddings
        self.logger.info("Creating node embeddings...")
        node_list = list(kg_graph.nodes())
        node_texts = []
        
        for node_id in node_list:
            node_data = kg_graph.nodes[node_id]
            node_type = node_data.get('type', '')
            node_name = node_data.get('id', str(node_id))
            
            if node_type == 'passage':
                text = node_data.get('text', node_name)[:500]
            else:
                text = node_name
            
            node_texts.append(text)
        
        node_embeddings = sentence_model.encode(
            node_texts,
            batch_size=self.config.retrieval.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.config.retrieval.normalize_embeddings
        )
        
        self.logger.info(f"Created embeddings for {len(node_list)} nodes")
        
        # Prepare graph data
        graph_data = {
            'KG': kg_graph,
            'text_dict': text_dict,
            'node_list': node_list,
            'node_embeddings': node_embeddings,
            'edge_list': list(kg_graph.edges())
        }
        
        # Create sentence encoder wrapper
        class SentenceEncoder:
            def __init__(self, model, normalize):
                self.model = model
                self.normalize = normalize
            
            def encode(self, texts, query_type=None):
                return self.model.encode(texts, normalize_embeddings=self.normalize, show_progress_bar=False)
        
        sentence_encoder = SentenceEncoder(
            sentence_model,
            self.config.retrieval.normalize_embeddings
        )
        
        # Create LLM client for generation
        self.logger.info("Initializing LLM client for answer generation...")
        generation_llm = LLMClient(
            api_key=self.config.get_api_key(),
            base_url=self.config.get_base_url(),
            model_name=self.config.get_generation_model(),
            timeout=self.config.api.timeout,
            max_retries=self.config.api.max_retries
        )
        
        # Initialize retrievers
        retriever_instances = {}
        
        if 'both' in retrievers or 'original' in retrievers:
            self.logger.info("Initializing HippoRAG (Original)...")
            retriever_instances['HippoRAG_Original'] = HippoRAGRetriever(
                llm_client=generation_llm,
                sentence_encoder=sentence_encoder,
                graph_data=graph_data,
                config=self.config.retrieval
            )
        
        if 'both' in retrievers or 'enhanced' in retrievers:
            self.logger.info("Initializing HippoRAG (Enhanced)...")
            retriever_instances['HippoRAG_Enhanced'] = HippoRAGEnhancedRetriever(
                llm_client=generation_llm,
                sentence_encoder=sentence_encoder,
                graph_data=graph_data,
                config=self.config.retrieval
            )
        
        # Create evaluator
        benchmark_dir = self.output_dir / "benchmark"
        evaluator = RAGEvaluator(
            config=self.config,
            llm_client=generation_llm,
            output_dir=str(benchmark_dir)
        )
        
        # Evaluate each retriever
        all_results = {
            'dataset': dataset_name,
            'num_samples': self.config.benchmark.num_samples,
            'graph_file': self.graph_file,
            'retrievers': {}
        }
        
        for retriever_name, retriever in retriever_instances.items():
            self.logger.info(f"\n{'=' * 80}")
            self.logger.info(f"Evaluating {retriever_name}")
            self.logger.info('=' * 80)
            
            results = evaluator.evaluate_retriever(
                retriever=retriever,
                retriever_name=retriever_name,
                dataset_name=dataset_name,
                passages_df=passages_df
            )
            
            all_results['retrievers'][retriever_name] = results
        
        # Save results
        evaluator.save_results(all_results, dataset_name)
        
        self.logger.info("Step 3 completed successfully!")
        
        return all_results
    
    def run_full_pipeline(
        self,
        dataset_name: str,
        max_samples: int = -1,
        num_eval_samples: int = None,
        retrievers: list = None,
        skip_existing: bool = False
    ) -> dict:
        """
        Run complete pipeline: extraction -> graph -> benchmark.
        
        Args:
            dataset_name: Dataset name
            max_samples: Max samples for extraction (-1 = all)
            num_eval_samples: Samples for evaluation (None = use max_samples)
            retrievers: Retrievers to test ['original', 'enhanced', 'both'] (default: ['enhanced'])
            skip_existing: Skip steps if outputs exist
            
        Returns:
            Dictionary with all results
        """
        self.logger.info("=" * 80)
        self.logger.info("TERAG COMPLETE PIPELINE")
        self.logger.info("=" * 80)
        self.logger.info(f"Dataset: {dataset_name}")
        self.logger.info(f"Extraction samples: {max_samples if max_samples > 0 else 'all'}")
        
        # Determine actual eval samples to be used
        actual_eval_samples = num_eval_samples if num_eval_samples is not None else (max_samples if max_samples > 0 else None)
        if actual_eval_samples is not None and actual_eval_samples > 0:
            self.logger.info(f"Evaluation samples: {actual_eval_samples}")
        elif num_eval_samples is None and max_samples > 0:
            self.logger.info(f"Evaluation samples: {max_samples} (same as extraction)")
        else:
            self.logger.info(f"Evaluation samples: all (from config)")
        
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("=" * 80)
        
        try:
            # Step 1: Concept Extraction
            self.step1_concept_extraction(
                dataset_name=dataset_name,
                max_samples=max_samples,
                skip_if_exists=skip_existing
            )
            
            # Step 2: Graph Building
            self.step2_graph_building(skip_if_exists=skip_existing)
            
            # Step 3: Benchmark Evaluation
            # If num_eval_samples not specified, use max_samples
            eval_samples_to_use = num_eval_samples if num_eval_samples is not None else (max_samples if max_samples > 0 else None)
            
            results = self.step3_benchmark_evaluation(
                dataset_name=dataset_name,
                num_samples=eval_samples_to_use,
                retrievers=retrievers
            )
            
            # Print final summary
            self.logger.info("\n" + "=" * 80)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 80)
            self.logger.info("\nGenerated Files:")
            self.logger.info(f"  Concepts CSV: {self.concepts_csv}")
            self.logger.info(f"  Passages CSV: {self.passages_csv}")
            self.logger.info(f"  Knowledge Graph: {self.graph_file}")
            self.logger.info(f"  Benchmark Results: {self.output_dir / 'benchmark'}")
            
            self.logger.info("\nEvaluation Summary:")
            for retriever_name, metrics in results['retrievers'].items():
                self.logger.info(f"\n{retriever_name}:")
                self.logger.info(f"  EM:        {metrics.get('avg_em', 0):.4f}")
                self.logger.info(f"  F1:        {metrics.get('avg_f1', 0):.4f}")
                self.logger.info(f"  Recall@2:  {metrics.get('avg_recall_at_2', 0):.4f}")
                self.logger.info(f"  Recall@5:  {metrics.get('avg_recall_at_5', 0):.4f}")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    parser = argparse.ArgumentParser(
        description='TERAG Complete Pipeline: Extraction -> Graph Building -> Benchmark'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    # Dataset settings
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['hotpotqa', '2wikimultihopqa', 'musique'],
        help='Dataset name'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=-1,
        help='Maximum samples for concept extraction (-1 = all)'
    )
    parser.add_argument(
        '--eval_samples',
        type=int,
        default=None,
        help='Number of samples for evaluation (default: same as max_samples)'
    )
    
    # Output settings
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output/pipeline',
        help='Base output directory for all results'
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help='Log file path (default: output_dir/pipeline.log)'
    )
    
    # Retriever settings
    parser.add_argument(
        '--retrievers',
        type=str,
        nargs='+',
        default=['enhanced'],
        choices=['original', 'enhanced', 'both'],
        help='Retrievers to evaluate (default: enhanced)'
    )
    
    # Pipeline control
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip steps if output files already exist'
    )
    parser.add_argument(
        '--step',
        type=int,
        choices=[1, 2, 3],
        default=None,
        help='Run only specific step (1=extraction, 2=graph, 3=benchmark)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.log_file is None:
        args.log_file = str(Path(args.output_dir) / "pipeline.log")
    
    setup_logging(log_level="INFO", log_file=args.log_file)
    
    # Load configuration
    logger = logging.getLogger(__name__)
    logger.info("Loading configuration...")
    config = ConfigLoader(args.config)
    
    # Create pipeline
    pipeline = TERAGPipeline(config=config, output_dir=args.output_dir)
    
    try:
        if args.step is None:
            # Run full pipeline
            pipeline.run_full_pipeline(
                dataset_name=args.dataset,
                max_samples=args.max_samples,
                num_eval_samples=args.eval_samples,
                retrievers=args.retrievers,
                skip_existing=args.skip_existing
            )
        else:
            # Run specific step
            if args.step == 1:
                pipeline.step1_concept_extraction(
                    dataset_name=args.dataset,
                    max_samples=args.max_samples,
                    skip_if_exists=args.skip_existing
                )
            elif args.step == 2:
                # Need to load existing CSV files
                csv_dir = Path(args.output_dir) / "extraction" / "concept_csv"
                concepts_files = list(csv_dir.glob("concepts_*.csv"))
                passages_files = list(csv_dir.glob("passages_*.csv"))
                
                if not concepts_files or not passages_files:
                    logger.error("No existing CSV files found. Run step 1 first.")
                    return 1
                
                pipeline.concepts_csv = str(concepts_files[-1])
                pipeline.passages_csv = str(passages_files[-1])
                pipeline.step2_graph_building(skip_if_exists=args.skip_existing)
            
            elif args.step == 3:
                # Need to load existing graph and CSV files
                graph_dir = Path(args.output_dir) / "graph"
                csv_dir = Path(args.output_dir) / "extraction" / "concept_csv"
                
                graph_files = list(graph_dir.glob("knowledge_graph_*.graphml"))
                passages_files = list(csv_dir.glob("passages_*.csv"))
                
                if not graph_files or not passages_files:
                    logger.error("No existing graph/passages files found. Run previous steps first.")
                    return 1
                
                pipeline.graph_file = str(graph_files[-1])
                pipeline.passages_csv = str(passages_files[-1])
                
                pipeline.step3_benchmark_evaluation(
                    dataset_name=args.dataset,
                    num_samples=args.eval_samples,
                    retrievers=args.retrievers
                )
        
        return 0
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

