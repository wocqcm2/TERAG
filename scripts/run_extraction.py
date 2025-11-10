#!/usr/bin/env python3
"""
Concept extraction script for TERAG framework.
Extracts named entities and document-level concepts from benchmark datasets.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from terag.utils.config_loader import ConfigLoader
from terag.utils.llm_client import LLMClient
from terag.extraction import ConceptExtractor


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def main():
    parser = argparse.ArgumentParser(
        description='Extract concepts from benchmark datasets'
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
        '--max_samples',
        type=int,
        default=-1,
        help='Maximum number of samples to process (-1 for all)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output/extraction',
        help='Output directory for results'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Path to benchmark data directory (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = ConfigLoader(args.config)
    setup_logging(config.general.log_level if hasattr(config, 'general') else "INFO")
    
    # Create LLM client for extraction
    print("Initializing LLM client...")
    extraction_llm = LLMClient(
        api_key=config.get_api_key(),
        base_url=config.get_base_url(),
        model_name=config.get_extraction_model(),
        timeout=config.api.timeout,
        max_retries=config.api.max_retries
    )
    
    # Create concept extractor
    print("Initializing concept extractor...")
    extractor = ConceptExtractor(
        config=config,
        llm_client=extraction_llm,
        output_dir=args.output_dir
    )
    
    # Determine data path
    data_path = args.data_path
    if data_path is None:
        # Use path from config
        if hasattr(config, 'benchmark') and config.benchmark.datasets:
            # Extract directory from dataset path
            dataset_file = config.benchmark.datasets.get(args.dataset)
            if dataset_file:
                data_path = str(Path(dataset_file).parent)
    
    if data_path is None:
        print("Error: Could not determine benchmark data path")
        print("Please specify --data_path or configure benchmark.datasets in config.yaml")
        return 1
    
    # Run extraction
    print("=" * 80)
    print(f"Starting concept extraction for {args.dataset}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max samples: {args.max_samples if args.max_samples > 0 else 'all'}")
    print("=" * 80)
    
    try:
        # Extract concepts
        concept_file = extractor.run_extraction(
            dataset_name=args.dataset,
            data_path=data_path,
            max_samples=args.max_samples
        )
        
        # Create CSV files
        print("\nCreating CSV files...")
        concepts_csv, passages_csv = extractor.create_csv_files(concept_file)
        
        # Aggregate usage statistics
        if config.extraction.record_usage:
            print("\nAggregating usage statistics...")
            usage_file, usage_summary = extractor.aggregate_usage_stats(concept_file)
        
        print("\n" + "=" * 80)
        print("Concept extraction completed successfully!")
        print("=" * 80)
        print(f"Concept file: {concept_file}")
        print(f"Concepts CSV: {concepts_csv}")
        print(f"Passages CSV: {passages_csv}")
        
        if config.extraction.record_usage:
            print(f"Usage statistics: {usage_file}")
            print(f"Total tokens used: {int(usage_summary['totals']['total_tokens'])}")
        
        return 0
    
    except Exception as e:
        print(f"\nError during extraction: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

