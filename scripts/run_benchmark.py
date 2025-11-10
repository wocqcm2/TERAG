#!/usr/bin/env python3
"""
Benchmark evaluation script for TERAG framework.
Evaluates RAG performance on multi-hop QA datasets.
"""

import argparse
import logging
import sys
import torch
import pandas as pd
import networkx as nx
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from terag.utils.config_loader import ConfigLoader
from terag.utils.llm_client import LLMClient
from terag.retrieval import HippoRAGRetriever, HippoRAGEnhancedRetriever
from terag.benchmark import RAGEvaluator


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_graph_and_embeddings(graph_file: str, passages_file: str, config: ConfigLoader):
    """
    Load graph and create embeddings.
    
    Args:
        graph_file: Path to GraphML file
        passages_file: Path to passages CSV file
        config: Configuration loader
        
    Returns:
        Dictionary with graph data and embeddings
    """
    print("Loading knowledge graph...")
    kg_graph = nx.read_graphml(graph_file)
    print(f"Loaded graph: {len(kg_graph.nodes)} nodes, {len(kg_graph.edges)} edges")
    
    print("Loading passages...")
    passages_df = pd.read_csv(passages_file)
    print(f"Loaded {len(passages_df)} passages")
    
    # Create text dictionary
    text_dict = {}
    for node_id, node_data in kg_graph.nodes(data=True):
        if node_data.get('type') == 'passage' and 'text' in node_data:
            text_dict[node_id] = node_data['text']
    
    print(f"Text dictionary: {len(text_dict)} entries")
    
    # Load embedding model
    print(f"Loading embedding model: {config.retrieval.embedding_model}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sentence_model = SentenceTransformer(config.retrieval.embedding_model, device=device)
    
    # Create embeddings for nodes
    print("Creating node embeddings...")
    node_list = list(kg_graph.nodes())
    node_texts = []
    
    for node_id in node_list:
        node_data = kg_graph.nodes[node_id]
        node_type = node_data.get('type', '')
        node_name = node_data.get('id', str(node_id))
        
        if node_type == 'passage':
            text = node_data.get('text', node_name)[:500]  # Limit length
        else:
            text = node_name
        
        node_texts.append(text)
    
    node_embeddings = sentence_model.encode(
        node_texts,
        batch_size=config.retrieval.embedding_batch_size,
        show_progress_bar=True,
        normalize_embeddings=config.retrieval.normalize_embeddings
    )
    
    print(f"Created embeddings for {len(node_list)} nodes")
    
    # Prepare data dictionary
    graph_data = {
        'KG': kg_graph,
        'text_dict': text_dict,
        'node_list': node_list,
        'node_embeddings': node_embeddings,
        'edge_list': list(kg_graph.edges())
    }
    
    # Add sentence encoder wrapper
    class SentenceEncoder:
        def __init__(self, model):
            self.model = model
        
        def encode(self, texts, query_type=None):
            return self.model.encode(texts, normalize_embeddings=config.retrieval.normalize_embeddings)
    
    sentence_encoder = SentenceEncoder(sentence_model)
    
    return graph_data, sentence_encoder, passages_df


def main():
    parser = argparse.ArgumentParser(
        description='Run RAG benchmark evaluation'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['hotpotqa', '2wikimultihopqa', 'musique'],
        help='Dataset name'
    )
    parser.add_argument(
        '--graph',
        type=str,
        required=True,
        help='Path to GraphML file'
    )
    parser.add_argument(
        '--passages',
        type=str,
        required=True,
        help='Path to passages CSV file'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of samples to evaluate (overrides config)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output/benchmark',
        help='Output directory for results'
    )
    parser.add_argument(
        '--retrievers',
        type=str,
        nargs='+',
        default=['original', 'enhanced'],
        choices=['original', 'enhanced', 'both'],
        help='Retrievers to evaluate'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = ConfigLoader(args.config)
    setup_logging(config.general.log_level if hasattr(config, 'general') else "INFO")
    
    # Override num_samples if specified
    if args.num_samples is not None:
        config.benchmark.num_samples = args.num_samples
    
    # Verify input files
    graph_file = Path(args.graph)
    passages_file = Path(args.passages)
    
    if not graph_file.exists():
        print(f"Error: Graph file not found: {graph_file}")
        return 1
    
    if not passages_file.exists():
        print(f"Error: Passages file not found: {passages_file}")
        return 1
    
    print("=" * 80)
    print("RAG Benchmark Evaluation")
    print(f"Dataset: {args.dataset}")
    print(f"Graph: {graph_file}")
    print(f"Passages: {passages_file}")
    print(f"Num samples: {config.benchmark.num_samples if config.benchmark.num_samples > 0 else 'all'}")
    print("=" * 80)
    
    try:
        # Load graph and create embeddings
        graph_data, sentence_encoder, passages_df = load_graph_and_embeddings(
            str(graph_file),
            str(passages_file),
            config
        )
        
        # Create LLM client for generation
        print("\nInitializing LLM client for answer generation...")
        generation_llm = LLMClient(
            api_key=config.get_api_key(),
            base_url=config.get_base_url(),
            model_name=config.get_generation_model(),
            timeout=config.api.timeout,
            max_retries=config.api.max_retries
        )
        
        # Create evaluator
        evaluator = RAGEvaluator(
            config=config,
            llm_client=generation_llm,
            output_dir=args.output_dir
        )
        
        # Initialize retrievers
        retrievers = {}
        
        if 'both' in args.retrievers or 'original' in args.retrievers:
            print("\nInitializing HippoRAG (Original)...")
            retrievers['HippoRAG_Original'] = HippoRAGRetriever(
                llm_client=generation_llm,
                sentence_encoder=sentence_encoder,
                graph_data=graph_data,
                config=config.retrieval
            )
        
        if 'both' in args.retrievers or 'enhanced' in args.retrievers:
            print("Initializing HippoRAG (Enhanced)...")
            retrievers['HippoRAG_Enhanced'] = HippoRAGEnhancedRetriever(
                llm_client=generation_llm,
                sentence_encoder=sentence_encoder,
                graph_data=graph_data,
                config=config.retrieval
            )
        
        # Evaluate each retriever
        all_results = {
            'dataset': args.dataset,
            'num_samples': config.benchmark.num_samples,
            'graph_file': str(graph_file),
            'retrievers': {}
        }
        
        for retriever_name, retriever in retrievers.items():
            print(f"\n{'=' * 80}")
            print(f"Evaluating {retriever_name}")
            print('=' * 80)
            
            results = evaluator.evaluate_retriever(
                retriever=retriever,
                retriever_name=retriever_name,
                dataset_name=args.dataset,
                passages_df=passages_df
            )
            
            all_results['retrievers'][retriever_name] = results
        
        # Save combined results
        evaluator.save_results(all_results, args.dataset)
        
        print("\n" + "=" * 80)
        print("Benchmark evaluation completed successfully!")
        print("=" * 80)
        
        # Print summary
        print("\nSummary:")
        for retriever_name, results in all_results['retrievers'].items():
            print(f"\n{retriever_name}:")
            print(f"  EM: {results.get('avg_em', 0):.4f}")
            print(f"  F1: {results.get('avg_f1', 0):.4f}")
            print(f"  Recall@2: {results.get('avg_recall_at_2', 0):.4f}")
            print(f"  Recall@5: {results.get('avg_recall_at_5', 0):.4f}")
        
        return 0
    
    except Exception as e:
        print(f"\nError during benchmark evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

