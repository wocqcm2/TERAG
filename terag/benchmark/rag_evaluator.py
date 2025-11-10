"""
RAG Evaluator for TERAG framework.
Evaluates retrieval and generation performance on multi-hop QA benchmarks.
"""

import json
import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from collections import Counter

from terag.utils.config_loader import ConfigLoader
from terag.utils.llm_client import LLMClient


logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculator for evaluation metrics."""
    
    @staticmethod
    def normalize_answer(answer: str) -> str:
        """Normalize answer for comparison."""
        import re
        
        # Extract answer after "Answer:" if present
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1]
        elif "answer:" in answer:
            answer = answer.split("answer:")[-1]
        
        if not answer:
            return "none"
        
        # Normalize: lowercase, remove punctuation, standardize whitespace
        answer = answer.lower()
        answer = answer.replace('-', ' ')
        answer = re.sub(r'[^\w\s]', '', answer)
        return ' '.join(answer.split())
    
    @staticmethod
    def calculate_em(prediction: str, ground_truth: str) -> float:
        """Calculate Exact Match score."""
        pred_normalized = MetricsCalculator.normalize_answer(prediction)
        gt_normalized = MetricsCalculator.normalize_answer(ground_truth)
        return 1.0 if pred_normalized == gt_normalized else 0.0
    
    @staticmethod
    def calculate_f1(prediction: str, ground_truth: str) -> float:
        """Calculate F1 score."""
        pred_normalized = MetricsCalculator.normalize_answer(prediction)
        gt_normalized = MetricsCalculator.normalize_answer(ground_truth)
        
        pred_tokens = pred_normalized.split()
        gt_tokens = gt_normalized.split()
        
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
        
        precision = num_same / len(pred_tokens) if pred_tokens else 0.0
        recall = num_same / len(gt_tokens) if gt_tokens else 0.0
        
        if (precision + recall) == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)


class RAGEvaluator:
    """
    RAG evaluator for multi-hop QA benchmarks.
    Evaluates retrieval and generation performance.
    """
    
    def __init__(
        self,
        config: ConfigLoader,
        llm_client: LLMClient,
        output_dir: str = "output"
    ):
        """
        Initialize RAG evaluator.
        
        Args:
            config: Configuration loader
            llm_client: LLM client for answer generation
            output_dir: Output directory for results
        """
        self.config = config
        self.llm_client = llm_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_calc = MetricsCalculator()
    
    def load_benchmark_data(self, dataset_name: str) -> List[Dict]:
        """
        Load benchmark dataset.
        
        Args:
            dataset_name: Name of dataset
            
        Returns:
            List of samples
        """
        dataset_path = self.config.benchmark.datasets.get(dataset_name)
        if not dataset_path:
            raise ValueError(f"Dataset {dataset_name} not found in config")
        
        logger.info(f"Loading benchmark data: {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Limit samples if configured
        num_samples = self.config.benchmark.num_samples
        if num_samples > 0 and len(data) > num_samples:
            data = data[:num_samples]
            logger.info(f"Using first {num_samples} samples")
        else:
            logger.info(f"Using all {len(data)} samples")
        
        return data
    
    def extract_supporting_facts(self, sample: Dict, dataset_name: str) -> List[str]:
        """
        Extract supporting fact titles from sample.
        
        Args:
            sample: Data sample
            dataset_name: Name of dataset
            
        Returns:
            List of supporting document titles
        """
        supporting_facts = []
        
        if dataset_name in ["hotpotqa", "2wikimultihopqa"]:
            for fact in sample.get("supporting_facts", []):
                supporting_facts.append(fact[0])  # Document title
        elif dataset_name == "musique":
            for paragraph in sample.get("paragraphs", []):
                if paragraph.get("is_supporting", False):
                    supporting_facts.append(paragraph.get("title", ""))
        
        return supporting_facts
    
    def calculate_recall_at_k(
        self,
        retrieved_ids: List[str],
        supporting_facts: List[str],
        passages_df: pd.DataFrame,
        k: int
    ) -> float:
        """
        Calculate Recall@K metric.
        
        Args:
            retrieved_ids: List of retrieved passage IDs
            supporting_facts: List of supporting document titles
            passages_df: DataFrame with passage information
            k: K value for recall
            
        Returns:
            Recall@K score
        """
        if not supporting_facts:
            return 0.0
        
        # Build title to passage_id mapping
        title_to_passage_id = {}
        for _, row in passages_df.iterrows():
            title = str(row['title']).strip()
            passage_id = str(row['passage_id']).strip()
            title_to_passage_id[title] = passage_id
        
        # Convert titles to passage_ids
        relevant_passage_ids = set()
        for title in supporting_facts:
            if title and isinstance(title, str):
                title_clean = title.strip()
                if title_clean in title_to_passage_id:
                    passage_id = title_to_passage_id[title_clean]
                    relevant_passage_ids.add(passage_id)
        
        if not relevant_passage_ids:
            return 0.0
        
        # Check how many relevant docs are in top-k
        found_relevant = set()
        for i in range(min(k, len(retrieved_ids))):
            doc_id = retrieved_ids[i]
            if doc_id in relevant_passage_ids:
                found_relevant.add(doc_id)
        
        recall = len(found_relevant) / len(relevant_passage_ids)
        return recall
    
    def truncate_context(
        self,
        retrieved_texts: List[str],
        max_total_chars: int = 1500,
        max_per_doc: int = 300
    ) -> str:
        """
        Truncate retrieved texts to fit context limits.
        
        Args:
            retrieved_texts: List of retrieved passage texts
            max_total_chars: Maximum total characters
            max_per_doc: Maximum characters per document
            
        Returns:
            Truncated context string
        """
        truncated_texts = []
        total_chars = 0
        
        for text in retrieved_texts:
            if total_chars >= max_total_chars:
                break
            
            # Truncate individual document
            truncated_text = text[:max_per_doc]
            if len(text) > max_per_doc:
                truncated_text += "..."
            
            if total_chars + len(truncated_text) <= max_total_chars:
                truncated_texts.append(truncated_text)
                total_chars += len(truncated_text) + 1
            else:
                # Add what fits
                remaining = max_total_chars - total_chars - 1
                if remaining > 50:
                    truncated_texts.append(text[:remaining] + "...")
                break
        
        return "\n".join(truncated_texts)
    
    def evaluate_retriever(
        self,
        retriever,
        retriever_name: str,
        dataset_name: str,
        passages_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Evaluate a single retriever.
        
        Args:
            retriever: Retriever object with retrieve() method
            retriever_name: Name of retriever
            dataset_name: Name of dataset
            passages_df: DataFrame with passage information
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating {retriever_name} on {dataset_name}...")
        
        # Load benchmark data
        benchmark_data = self.load_benchmark_data(dataset_name)
        
        # Metrics storage
        em_scores = []
        f1_scores = []
        recall_2_scores = []
        recall_5_scores = []
        retrieval_times = []
        
        # Evaluate each sample
        for i, sample in enumerate(tqdm(benchmark_data, desc=f"Testing {retriever_name}")):
            question = sample['question']
            ground_truth = sample['answer']
            
            try:
                # Retrieve passages
                start_time = time.time()
                retrieved_texts, retrieved_ids = retriever.retrieve(
                    question,
                    topk=self.config.benchmark.topk_retrieval
                )
                retrieval_time = time.time() - start_time
                retrieval_times.append(retrieval_time)
                
                # Calculate recall
                supporting_facts = self.extract_supporting_facts(sample, dataset_name)
                
                if self.config.benchmark.compute_recall_at_2:
                    recall_2 = self.calculate_recall_at_k(
                        retrieved_ids, supporting_facts, passages_df, 2
                    )
                    recall_2_scores.append(recall_2)
                
                if self.config.benchmark.compute_recall_at_5:
                    recall_5 = self.calculate_recall_at_k(
                        retrieved_ids, supporting_facts, passages_df, 5
                    )
                    recall_5_scores.append(recall_5)
                
                # Generate answer
                context = self.truncate_context(
                    retrieved_texts,
                    max_total_chars=self.config.benchmark.max_context_length,
                    max_per_doc=self.config.benchmark.max_per_document
                )
                
                prediction = self.llm_client.generate_with_context(
                    question,
                    context,
                    max_new_tokens=self.config.benchmark.generation_max_tokens,
                    temperature=self.config.benchmark.generation_temperature
                )
                
                # Calculate metrics
                if self.config.benchmark.compute_em:
                    em = self.metrics_calc.calculate_em(prediction, ground_truth)
                    em_scores.append(em)
                
                if self.config.benchmark.compute_f1:
                    f1 = self.metrics_calc.calculate_f1(prediction, ground_truth)
                    f1_scores.append(f1)
            
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                em_scores.append(0.0)
                f1_scores.append(0.0)
                recall_2_scores.append(0.0)
                recall_5_scores.append(0.0)
                retrieval_times.append(0.0)
        
        # Aggregate results
        results = {
            'retriever_name': retriever_name,
            'dataset_name': dataset_name,
            'num_samples': len(benchmark_data),
            'avg_retrieval_time': np.mean(retrieval_times) if retrieval_times else 0.0
        }
        
        if self.config.benchmark.compute_em:
            results['avg_em'] = np.mean(em_scores)
        if self.config.benchmark.compute_f1:
            results['avg_f1'] = np.mean(f1_scores)
        if self.config.benchmark.compute_recall_at_2:
            results['avg_recall_at_2'] = np.mean(recall_2_scores)
        if self.config.benchmark.compute_recall_at_5:
            results['avg_recall_at_5'] = np.mean(recall_5_scores)
        
        logger.info(f"{retriever_name} results:")
        logger.info(f"  EM: {results.get('avg_em', 0):.4f}")
        logger.info(f"  F1: {results.get('avg_f1', 0):.4f}")
        logger.info(f"  Recall@2: {results.get('avg_recall_at_2', 0):.4f}")
        logger.info(f"  Recall@5: {results.get('avg_recall_at_5', 0):.4f}")
        
        return results
    
    def save_results(
        self,
        results: Dict[str, Any],
        dataset_name: str
    ):
        """
        Save evaluation results.
        
        Args:
            results: Results dictionary
            dataset_name: Name of dataset
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON results
        if self.config.benchmark.save_detailed_results:
            results_file = self.output_dir / f"results_{dataset_name}_{timestamp}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {results_file}")
        
        # Save summary text
        if self.config.benchmark.save_summary:
            summary_file = self.output_dir / f"summary_{dataset_name}_{timestamp}.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"RAG Evaluation Summary - {dataset_name}\n")
                f.write("=" * 80 + "\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Samples: {results.get('num_samples', 0)}\n")
                f.write("\n")
                
                for retriever_name, metrics in results.items():
                    if isinstance(metrics, dict) and 'avg_em' in metrics:
                        f.write(f"\n{retriever_name}:\n")
                        f.write(f"  EM: {metrics.get('avg_em', 0):.4f}\n")
                        f.write(f"  F1: {metrics.get('avg_f1', 0):.4f}\n")
                        f.write(f"  Recall@2: {metrics.get('avg_recall_at_2', 0):.4f}\n")
                        f.write(f"  Recall@5: {metrics.get('avg_recall_at_5', 0):.4f}\n")
                        f.write(f"  Avg retrieval time: {metrics.get('avg_retrieval_time', 0):.4f}s\n")
            
            logger.info(f"Summary saved to: {summary_file}")

