"""
Knowledge graph builder for TERAG framework.
Constructs knowledge graphs from extracted concepts with co-occurrence relationships.
"""

import json
import logging
import pandas as pd
import networkx as nx
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Any, Optional

from terag.utils.config_loader import ConfigLoader


logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Knowledge graph builder.
    Creates graphs with concept and passage nodes, plus co-occurrence edges.
    """
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize graph builder.
        
        Args:
            config: Configuration loader
        """
        self.config = config
        self.graph = nx.DiGraph()  # Directed graph
        self.concepts_df = None
        self.passages_df = None
        self.node_counter = 0
    
    def load_data(self, concepts_file: str, passages_file: str):
        """
        Load concept and passage data from CSV files.
        
        Args:
            concepts_file: Path to concepts CSV
            passages_file: Path to passages CSV
        """
        logger.info("Loading data...")
        
        # Load concepts
        self.concepts_df = pd.read_csv(concepts_file)
        logger.info(f"Loaded {len(self.concepts_df)} concepts")
        
        # Load passages
        self.passages_df = pd.read_csv(passages_file)
        logger.info(f"Loaded {len(self.passages_df)} passages")
        
        # Preprocessing
        self.concepts_df = self.concepts_df.fillna("")
        self.passages_df = self.passages_df.fillna("")
        
        logger.info("Data loading complete")
    
    def create_passage_nodes(self):
        """Create passage nodes in the graph."""
        if not self.config.graph.create_passage_nodes:
            logger.info("Skipping passage node creation (disabled in config)")
            return
        
        logger.info("Creating passage nodes...")
        
        for _, passage in tqdm(self.passages_df.iterrows(), total=len(self.passages_df), desc="Passages"):
            passage_id = str(passage['passage_id'])
            
            self.graph.add_node(
                passage_id,
                type="passage",
                id=passage['text'],  # Text content in 'id' attribute
                passage_id=passage_id,
                title=passage.get('title', ''),
                text=passage['text'],
                source_doc=passage['source_doc'],
                chunk_id=passage['chunk_id'],
                word_count=passage.get('word_count', 0),
                file_id=passage_id
            )
        
        logger.info(f"Created {len(self.passages_df)} passage nodes")
    
    def create_concept_nodes(self):
        """Create concept nodes in the graph."""
        if not self.config.graph.create_concept_nodes:
            logger.info("Skipping concept node creation (disabled in config)")
            return
        
        logger.info("Creating concept nodes...")
        
        # Get unique concepts
        unique_concepts = self.concepts_df.drop_duplicates(subset=['name', 'type'])
        logger.info(f"Found {len(unique_concepts)} unique concepts")
        
        # Pre-compute concept groups for efficiency
        concept_groups = self.concepts_df.groupby(['name', 'type'])
        
        for _, concept in tqdm(unique_concepts.iterrows(), total=len(unique_concepts), desc="Concepts"):
            node_id = f"{concept['type']}_{concept['concept_id']}"
            
            # Get all occurrences of this concept
            same_concepts = concept_groups.get_group((concept['name'], concept['type']))
            
            source_passages = same_concepts['passage_id'].unique().tolist()
            source_docs = same_concepts['source_doc'].unique().tolist()
            
            # Create file_id from passage_ids
            file_id = ','.join([str(pid) for pid in source_passages[:100]])
            
            # Node attributes
            node_attrs = {
                'type': concept['type'],
                'id': concept['name'],
                'source_passages': source_passages[:100],
                'source_docs': list(set(source_docs))[:50],
                'occurrence_count': len(same_concepts),
                'file_id': file_id
            }
            
            self.graph.add_node(node_id, **node_attrs)
        
        logger.info(f"Created {len(unique_concepts)} concept nodes")
    
    def create_has_passage_edges(self):
        """Create edges from concepts to passages (has_passage relation)."""
        if not self.config.graph.create_has_passage_edges:
            logger.info("Skipping has_passage edge creation (disabled in config)")
            return
        
        logger.info("Creating has_passage edges...")
        
        edge_count = 0
        for _, concept in tqdm(self.concepts_df.iterrows(), total=len(self.concepts_df), desc="Has_passage edges"):
            concept_node_id = f"{concept['type']}_{concept['concept_id']}"
            passage_node_id = str(concept['passage_id'])
            
            if self.graph.has_node(concept_node_id) and self.graph.has_node(passage_node_id):
                self.graph.add_edge(
                    concept_node_id,
                    passage_node_id,
                    relation="has_passage"
                )
                edge_count += 1
        
        logger.info(f"Created {edge_count} has_passage edges")
    
    def create_cooccurrence_edges(self):
        """Create co-occurrence edges between concepts in the same passage."""
        if not self.config.graph.create_cooccur_edges:
            logger.info("Skipping co-occurrence edge creation (disabled in config)")
            return
        
        logger.info("Creating co-occurrence edges...")
        
        # Expand passage_ids (handle comma-separated values)
        expanded_concepts = []
        for _, row in self.concepts_df.iterrows():
            passage_ids = str(row['passage_id']).split(',')
            for passage_id in passage_ids:
                passage_id = passage_id.strip()
                if passage_id:
                    row_copy = row.copy()
                    row_copy['passage_id'] = passage_id
                    expanded_concepts.append(row_copy)
        
        expanded_df = pd.DataFrame(expanded_concepts)
        logger.info(f"Expanded passage_id format: {len(self.concepts_df)} -> {len(expanded_df)} concept-passage relations")
        
        # Group by passage
        passage_groups = expanded_df.groupby('passage_id')
        
        cooccur_edge_count = 0
        max_concepts = self.config.graph.max_concepts_per_passage
        bidirectional = self.config.graph.cooccur_bidirectional
        
        for passage_id, group in tqdm(passage_groups, desc="Co-occur edges"):
            unique_concepts_in_passage = group.drop_duplicates(subset=['name', 'type'])
            concepts_list = [
                (f"{row['type']}_{row['concept_id']}", row['name'])
                for _, row in unique_concepts_in_passage.iterrows()
            ]
            
            # Limit concepts per passage
            if len(concepts_list) > max_concepts:
                concepts_list = concepts_list[:max_concepts]
            
            # Create co-occurrence edges
            for i in range(len(concepts_list)):
                for j in range(i + 1, len(concepts_list)):
                    node1, name1 = concepts_list[i]
                    node2, name2 = concepts_list[j]
                    
                    if (self.graph.has_node(node1) and self.graph.has_node(node2) and
                        not self.graph.has_edge(node1, node2) and
                        not self.graph.has_edge(node2, node1)):
                        
                        # Create edge(s)
                        self.graph.add_edge(
                            node1, node2,
                            relation="co_occur",
                            passage_id=passage_id
                        )
                        cooccur_edge_count += 1
                        
                        if bidirectional:
                            self.graph.add_edge(
                                node2, node1,
                                relation="co_occur",
                                passage_id=passage_id
                            )
                            cooccur_edge_count += 1
        
        logger.info(f"Created {cooccur_edge_count} co-occurrence edges")
    
    def build_graph(self, concepts_file: str, passages_file: str):
        """
        Build complete knowledge graph.
        
        Args:
            concepts_file: Path to concepts CSV
            passages_file: Path to passages CSV
        """
        logger.info("=" * 80)
        logger.info("Building knowledge graph...")
        logger.info("=" * 80)
        
        # Load data
        self.load_data(concepts_file, passages_file)
        
        # Create nodes
        self.create_passage_nodes()
        self.create_concept_nodes()
        
        # Create edges
        self.create_has_passage_edges()
        self.create_cooccurrence_edges()
        
        logger.info(f"Graph construction complete: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        logger.info("=" * 80)
    
    def export_to_graphml(self, output_file: str):
        """
        Export graph to GraphML format.
        
        Args:
            output_file: Path to output GraphML file
        """
        logger.info(f"Exporting graph to GraphML: {output_file}")
        
        # Create GraphML root
        root = ET.Element("graphml")
        root.set("xmlns", "http://graphml.graphdrawing.org/xmlns")
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set("xsi:schemaLocation", "http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd")
        
        # Define attribute keys
        keys = [
            ("type", "node", "type", "string"),
            ("id", "node", "id", "string"),
            ("file_id", "node", "file_id", "string"),
            ("text", "node", "text", "string"),
            ("title", "node", "title", "string"),
            ("source_doc", "node", "source_doc", "string"),
            ("occurrence_count", "node", "occurrence_count", "int"),
            ("chunk_id", "node", "chunk_id", "string"),
            ("word_count", "node", "word_count", "int"),
            ("source_passages", "node", "source_passages", "string"),
            ("source_docs", "node", "source_docs", "string"),
            ("relation", "edge", "relation", "string"),
            ("passage_id", "edge", "passage_id", "string"),
        ]
        
        for key_id, for_type, attr_name, attr_type in keys:
            key_elem = ET.SubElement(root, "key")
            key_elem.set("id", key_id)
            key_elem.set("for", for_type)
            key_elem.set("attr.name", attr_name)
            key_elem.set("attr.type", attr_type)
        
        # Create graph element
        graph_elem = ET.SubElement(root, "graph")
        graph_elem.set("id", "knowledge_graph")
        graph_elem.set("edgedefault", "directed")
        
        # Add nodes
        valid_node_attrs = ["type", "id", "text", "title", "source_doc",
                           "occurrence_count", "chunk_id", "file_id", 
                           "word_count", "source_passages", "source_docs"]
        
        for node_id, node_data in self.graph.nodes(data=True):
            node_elem = ET.SubElement(graph_elem, "node")
            node_elem.set("id", node_id)
            
            for attr_name, value in node_data.items():
                if attr_name in valid_node_attrs:
                    data_elem = ET.SubElement(node_elem, "data")
                    data_elem.set("key", attr_name)
                    
                    if value is None:
                        data_elem.text = ""
                    elif isinstance(value, (list, dict)):
                        data_elem.text = str(value)[:500]
                    else:
                        data_elem.text = str(value)
        
        # Add edges
        valid_edge_attrs = ["relation", "passage_id"]
        edge_id = 0
        
        for source, target, edge_data in self.graph.edges(data=True):
            edge_elem = ET.SubElement(graph_elem, "edge")
            edge_elem.set("id", f"e{edge_id}")
            edge_elem.set("source", source)
            edge_elem.set("target", target)
            edge_id += 1
            
            for attr_name, value in edge_data.items():
                if attr_name in valid_edge_attrs:
                    data_elem = ET.SubElement(edge_elem, "data")
                    data_elem.set("key", attr_name)
                    data_elem.text = str(value) if value is not None else ""
        
        # Pretty print and save
        rough_string = ET.tostring(root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
        
        logger.info(f"GraphML file saved: {output_file}")
    
    def print_statistics(self, output_file: Optional[str] = None):
        """
        Print and optionally save graph statistics.
        
        Args:
            output_file: Optional path to save statistics JSON
        """
        stats_data = {}
        
        logger.info("=" * 80)
        logger.info("Knowledge Graph Statistics")
        logger.info("=" * 80)
        logger.info(f"Total nodes: {self.graph.number_of_nodes()}")
        logger.info(f"Total edges: {self.graph.number_of_edges()}")
        
        # Basic stats
        stats_data['basic_stats'] = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Node type distribution
        node_types = defaultdict(int)
        for _, data in self.graph.nodes(data=True):
            node_types[data.get('type', 'unknown')] += 1
        
        stats_data['node_types'] = dict(node_types)
        
        logger.info("\nNode type distribution:")
        for node_type, count in node_types.items():
            logger.info(f"  {node_type}: {count}")
        
        # Edge type distribution
        edge_types = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            edge_types[data.get('relation', 'unknown')] += 1
        
        stats_data['edge_types'] = dict(edge_types)
        
        logger.info("\nEdge type distribution:")
        for edge_type, count in edge_types.items():
            logger.info(f"  {edge_type}: {count}")
        
        # Node type connections
        logger.info("\nNode type connections (source -> target):")
        connection_stats = defaultdict(lambda: defaultdict(int))
        
        for source, target, edge_data in self.graph.edges(data=True):
            source_type = self.graph.nodes[source].get('type', 'unknown')
            target_type = self.graph.nodes[target].get('type', 'unknown')
            connection_stats[source_type][target_type] += 1
        
        stats_data['node_type_connections'] = {}
        for source_type in connection_stats:
            stats_data['node_type_connections'][source_type] = dict(connection_stats[source_type])
            for target_type in connection_stats[source_type]:
                count = connection_stats[source_type][target_type]
                logger.info(f"  {source_type} -> {target_type}: {count}")
        
        # Degree statistics
        logger.info("\nDegree statistics:")
        in_degrees = dict(self.graph.in_degree())
        out_degrees = dict(self.graph.out_degree())
        
        type_in_degrees = defaultdict(list)
        type_out_degrees = defaultdict(list)
        
        for node_id, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            type_in_degrees[node_type].append(in_degrees[node_id])
            type_out_degrees[node_type].append(out_degrees[node_id])
        
        stats_data['degree_stats'] = {}
        
        for node_type in sorted(type_in_degrees.keys()):
            avg_in = sum(type_in_degrees[node_type]) / len(type_in_degrees[node_type])
            avg_out = sum(type_out_degrees[node_type]) / len(type_out_degrees[node_type])
            max_in = max(type_in_degrees[node_type])
            max_out = max(type_out_degrees[node_type])
            
            stats_data['degree_stats'][node_type] = {
                'avg_in_degree': round(avg_in, 2),
                'avg_out_degree': round(avg_out, 2),
                'max_in_degree': max_in,
                'max_out_degree': max_out
            }
            
            logger.info(f"  {node_type}:")
            logger.info(f"    Average in-degree: {avg_in:.2f}, Max: {max_in}")
            logger.info(f"    Average out-degree: {avg_out:.2f}, Max: {max_out}")
        
        # Connectivity
        logger.info("\nConnectivity:")
        if self.graph.number_of_nodes() > 0:
            undirected_graph = self.graph.to_undirected()
            connected_components = list(nx.connected_components(undirected_graph))
            
            stats_data['connectivity_stats'] = {
                'connected_components_count': len(connected_components),
                'largest_component_size': max(len(comp) for comp in connected_components),
                'largest_component_ratio': max(len(comp) for comp in connected_components) / self.graph.number_of_nodes() * 100
            }
            
            logger.info(f"  Connected components: {len(connected_components)}")
            logger.info(f"  Largest component: {stats_data['connectivity_stats']['largest_component_size']} nodes ({stats_data['connectivity_stats']['largest_component_ratio']:.2f}%)")
        
        logger.info("=" * 80)
        
        # Save statistics if output file provided
        if output_file:
            stats_file = str(Path(output_file).with_suffix('.stats.json'))
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Statistics saved to: {stats_file}")

