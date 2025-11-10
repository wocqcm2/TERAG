#!/usr/bin/env python3
"""
Graph construction script for TERAG framework.
Builds knowledge graphs from extracted concepts.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from terag.utils.config_loader import ConfigLoader
from terag.graph import GraphBuilder


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def main():
    parser = argparse.ArgumentParser(
        description='Build knowledge graph from extracted concepts'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--concepts',
        type=str,
        required=True,
        help='Path to concepts CSV file'
    )
    parser.add_argument(
        '--passages',
        type=str,
        required=True,
        help='Path to passages CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for GraphML file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = ConfigLoader(args.config)
    setup_logging(config.general.log_level if hasattr(config, 'general') else "INFO")
    
    # Verify input files exist
    concepts_file = Path(args.concepts)
    passages_file = Path(args.passages)
    
    if not concepts_file.exists():
        print(f"Error: Concepts file not found: {concepts_file}")
        return 1
    
    if not passages_file.exists():
        print(f"Error: Passages file not found: {passages_file}")
        return 1
    
    # Create graph builder
    print("Initializing graph builder...")
    builder = GraphBuilder(config)
    
    # Build graph
    print("=" * 80)
    print("Building knowledge graph")
    print(f"Concepts file: {concepts_file}")
    print(f"Passages file: {passages_file}")
    print(f"Output file: {args.output}")
    print("=" * 80)
    
    try:
        # Build the graph
        builder.build_graph(
            concepts_file=str(concepts_file),
            passages_file=str(passages_file)
        )
        
        # Export to GraphML
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        builder.export_to_graphml(str(output_path))
        
        # Print statistics
        builder.print_statistics(str(output_path))
        
        print("\n" + "=" * 80)
        print("Graph construction completed successfully!")
        print("=" * 80)
        print(f"GraphML file: {output_path}")
        print(f"Statistics file: {output_path.with_suffix('.stats.json')}")
        
        return 0
    
    except Exception as e:
        print(f"\nError during graph construction: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

