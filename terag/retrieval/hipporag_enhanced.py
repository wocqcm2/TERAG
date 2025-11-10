"""
HippoRAG Enhanced Retriever.
Enhanced version with similarity-weighted personalization and configurable PageRank.
"""

import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

from terag.utils.llm_client import LLMClient
from terag.utils.config_loader import RetrievalConfig


class HippoRAGEnhancedRetriever:
    """
    HippoRAG retriever (enhanced version).
    Uses semantic similarity weights and improved entity matching.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        sentence_encoder,
        graph_data: Dict[str, Any],
        config: RetrievalConfig
    ):
        """
        Initialize enhanced HippoRAG retriever.
        
        Args:
            llm_client: LLM client for entity extraction
            sentence_encoder: Sentence embedding model
            graph_data: Graph data dictionary with KG, embeddings, etc.
            config: Retrieval configuration
        """
        self.llm_client = llm_client
        self.sentence_encoder = sentence_encoder
        self.config = config
        
        # Load graph data
        self.passage_dict = graph_data["text_dict"]
        self.node_embeddings = graph_data["node_embeddings"]
        if isinstance(self.node_embeddings, list):
            self.node_embeddings = np.array(self.node_embeddings)
        
        self.node_list = graph_data["node_list"]
        self.KG: nx.DiGraph = graph_data["KG"]
        
        # Build file_id mappings
        self.file_id_to_node_id = {}
        
        print("Building file_id mappings...")
        for node_id in tqdm(list(self.KG.nodes)):
            if self.KG.nodes[node_id]['type'] == "passage":
                file_ids = self.KG.nodes[node_id]['file_id'].split(',')
                for file_id in file_ids:
                    if file_id not in self.file_id_to_node_id:
                        self.file_id_to_node_id[file_id] = []
                    self.file_id_to_node_id[file_id].append(node_id)
        
        # Create subgraph with valid nodes
        self.KG = self.KG.subgraph(self.node_list)
        self.node_name_list = [self.KG.nodes[node]["id"] for node in self.node_list]
        
        print(f"HippoRAG (Enhanced) initialized: {len(self.node_list)} nodes")
    
    def retrieve_personalization_dict(
        self,
        query: str,
        topN: int = 10
    ) -> Dict[str, float]:
        """
        Build personalization dictionary with semantic similarity weights.
        
        Args:
            query: Query string
            topN: Number of top nodes to use
            
        Returns:
            Personalization dictionary mapping node IDs to weights
        """
        # Extract and normalize entities
        entities_str = self.llm_client.ner(query)
        entities = [e.strip() for e in entities_str.split(",") if e.strip()]
        
        # Normalize entity names
        normalized_entities = []
        for entity in entities:
            normalized = entity.strip().lower()
            
            # Remove garbage prefixes
            garbage_prefixes = ['entities:', 'entity:', 'entities :', 'entity :']
            for prefix in garbage_prefixes:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):].strip()
                    break
            
            # Filter empty or too short
            if normalized and len(normalized) >= 2:
                normalized_entities.append(normalized)
        
        entities = normalized_entities if normalized_entities else [query.lower()]
        
        if len(entities) == 0:
            entities = [query.lower()]
        
        # Evenly distribute topk for each entity
        topk_for_each_entity = topN // len(entities)
        
        # Retrieve top k nodes with similarity scores
        topk_nodes = []
        node_similarity_scores = {}
        
        for entity in entities:
            # Normalize node names for matching
            normalized_node_names = [name.lower() for name in self.node_name_list]
            
            if entity in normalized_node_names:
                # Exact match
                index = normalized_node_names.index(entity)
                node = self.node_list[index]
                topk_nodes.append(node)
                node_similarity_scores[node] = 1.0  # Perfect match
            else:
                # Semantic search - only top 1
                topk_for_this_entity = 1
                
                entity_embedding = self.sentence_encoder.encode([entity], query_type="search")
                scores = self.node_embeddings @ entity_embedding[0].T
                index_matrix = np.argsort(scores)[-topk_for_this_entity:][::-1]
                
                for i in index_matrix:
                    node = self.node_list[i]
                    topk_nodes.append(node)
                    node_similarity_scores[node] = float(scores[i])
        
        topk_nodes = list(set(topk_nodes))
        
        # Limit to 2*topN
        if len(topk_nodes) > 2 * topN:
            topk_nodes = topk_nodes[:2 * topN]
        
        # Calculate frequency-based weights
        freq_dict_for_nodes = {}
        for node in topk_nodes:
            node_data = self.KG.nodes[node]
            file_ids = node_data["file_id"]
            file_ids_list = file_ids.split(",")
            file_ids_list = list(set(file_ids_list))
            freq_dict_for_nodes[node] = len(file_ids_list)
        
        # Enhanced personalization dict: similarity * inverse frequency
        personalization_dict = {}
        for node in topk_nodes:
            similarity_score = node_similarity_scores.get(node, 0.0)
            freq_inverse = 1 / freq_dict_for_nodes[node]
            combined_weight = similarity_score * freq_inverse
            personalization_dict[node] = combined_weight
        
        return personalization_dict
    
    def retrieve(
        self,
        query: str,
        topk: int = 5
    ) -> Tuple[List[str], List[str]]:
        """
        Retrieve top-k passages for a query.
        
        Args:
            query: Query string
            topk: Number of passages to retrieve
            
        Returns:
            Tuple of (passage_contents, passage_ids)
        """
        topk_nodes = self.config.enhanced_topk_nodes
        personalization_dict = self.retrieve_personalization_dict(query, topN=topk_nodes)
        
        # Run PageRank with enhanced parameters
        pr = nx.pagerank(
            self.KG,
            personalization=personalization_dict,
            alpha=self.config.enhanced_ppr_alpha,
            max_iter=self.config.enhanced_ppr_max_iter,
            tol=self.config.enhanced_ppr_tol
        )
        
        # Round and filter low scores
        for node in pr:
            pr[node] = round(pr[node], 4)
            if pr[node] < 0.001:
                pr[node] = 0
        
        # Aggregate passage scores
        passage_probabilities_sum = {}
        for node in pr:
            node_data = self.KG.nodes[node]
            file_ids = node_data["file_id"]
            file_ids_list = file_ids.split(",")
            file_ids_list = list(set(file_ids_list))
            
            for file_id in file_ids_list:
                if file_id == 'concept_file':
                    continue
                if file_id not in self.file_id_to_node_id:
                    continue
                
                for node_id in self.file_id_to_node_id[file_id]:
                    if node_id not in passage_probabilities_sum:
                        passage_probabilities_sum[node_id] = 0
                    passage_probabilities_sum[node_id] += pr[node]
        
        # Sort and get top passages
        sorted_passages = sorted(
            passage_probabilities_sum.items(),
            key=lambda x: x[1],
            reverse=True
        )
        top_passages = sorted_passages[:topk]
        top_passage_ids, scores = zip(*top_passages) if top_passages else ([], [])
        
        # Get passage contents
        passage_contents = [
            self.passage_dict[passage_id]
            for passage_id in top_passage_ids
        ]
        
        return passage_contents, list(top_passage_ids)

