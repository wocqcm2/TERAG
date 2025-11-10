"""
HippoRAG Original Retriever.
Original implementation from the HippoRAG paper.
"""

import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

from terag.utils.llm_client import LLMClient
from terag.utils.config_loader import RetrievalConfig


class HippoRAGRetriever:
    """
    HippoRAG retriever (original version).
    Uses named entity recognition and PageRank for passage retrieval.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        sentence_encoder,
        graph_data: Dict[str, Any],
        config: RetrievalConfig
    ):
        """
        Initialize HippoRAG retriever.
        
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
        self.node_id_to_file_id = {}
        
        print("Building file_id mappings...")
        for node_id in tqdm(list(self.KG.nodes)):
            if self.KG.nodes[node_id]['type'] == "passage":
                file_ids = self.KG.nodes[node_id]['file_id'].split(',')
                for file_id in file_ids:
                    if file_id not in self.file_id_to_node_id:
                        self.file_id_to_node_id[file_id] = []
                    self.file_id_to_node_id[file_id].append(node_id)
                    self.node_id_to_file_id[node_id] = file_id
        
        # Create subgraph with valid nodes
        self.KG = self.KG.subgraph(self.node_list)
        self.node_name_list = [self.KG.nodes[node]["id"] for node in self.node_list]
        
        print(f"HippoRAG (Original) initialized: {len(self.node_list)} nodes")
    
    def retrieve_personalization_dict(
        self,
        query: str,
        topN: int = 10
    ) -> Dict[str, float]:
        """
        Build personalization dictionary for PageRank.
        
        Args:
            query: Query string
            topN: Number of top nodes to use
            
        Returns:
            Personalization dictionary mapping node IDs to weights
        """
        # Extract entities from query
        entities_str = self.llm_client.ner(query)
        entities = [e.strip() for e in entities_str.split(",") if e.strip()]
        
        if len(entities) == 0:
            entities = [query]
        
        # Evenly distribute topk for each entity
        topk_for_each_entity = topN // len(entities)
        
        # Retrieve top k nodes
        topk_nodes = []
        
        for entity in entities:
            if entity in self.node_name_list:
                # Exact match
                index = self.node_name_list.index(entity)
                topk_nodes.append(self.node_list[index])
            else:
                # Semantic search
                topk_for_this_entity = 1
                
                entity_embedding = self.sentence_encoder.encode([entity], query_type="search")
                scores = self.node_embeddings @ entity_embedding[0].T
                index_matrix = np.argsort(scores)[-topk_for_this_entity:][::-1]
                
                topk_nodes += [self.node_list[i] for i in index_matrix]
        
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
        
        # Personalization dict: inverse frequency
        personalization_dict = {
            node: 1 / freq_dict_for_nodes[node]
            for node in topk_nodes
        }
        
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
        topk_nodes = self.config.original_topk_nodes
        personalization_dict = self.retrieve_personalization_dict(query, topN=topk_nodes)
        
        # Run PageRank
        pr = nx.pagerank(
            self.KG,
            personalization=personalization_dict,
            alpha=self.config.original_ppr_alpha,
            max_iter=self.config.original_ppr_max_iter,
            tol=self.config.original_ppr_tol
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
        
        # Convert to file_ids for recall calculation
        top_file_ids = [
            self.node_id_to_file_id.get(passage_id, passage_id)
            for passage_id in top_passage_ids
        ]
        
        return passage_contents, list(top_passage_ids)

