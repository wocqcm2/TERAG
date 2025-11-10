"""
Concept extractor for TERAG framework.
Extracts named entities and document-level concepts from text passages.
"""

import os
import json
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm

from terag.utils.config_loader import ConfigLoader
from terag.utils.llm_client import LLMClient
from terag.extraction.data_processor import DataProcessor


class ConceptOutputParser:
    """
    Parser for concept extraction LLM outputs.
    Parses structured outputs into concept dictionaries.
    """
    
    def __init__(
        self,
        normalize_names: bool = True,
        filter_low_quality: bool = True
    ):
        """
        Initialize output parser.
        
        Args:
            normalize_names: Whether to normalize concept names
            filter_low_quality: Whether to filter low-quality concepts
        """
        self.normalize_names = normalize_names
        self.filter_low_quality = filter_low_quality
    
    def normalize_concept_name(self, name: str) -> str:
        """
        Normalize concept name.
        
        Args:
            name: Raw concept name
            
        Returns:
            Normalized concept name
        """
        if not self.normalize_names or not name or not isinstance(name, str):
            return name
        
        # Basic normalization
        normalized = name.strip().lower()
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'[^\w\s\-\.\,\(\)\/]', '', normalized)
        
        # Limit length
        if len(normalized) > 100:
            normalized = normalized[:97] + "..."
        
        # Filter empty results
        if len(normalized.strip()) < 2:
            return ""
        
        return normalized
    
    def is_low_quality_concept(self, name: str) -> bool:
        """
        Check if concept is low quality.
        
        Args:
            name: Concept name
            
        Returns:
            True if low quality
        """
        if not self.filter_low_quality:
            return False
        
        if not name or len(name.strip()) < 2:
            return True
        
        # Filter pure numbers
        if name.strip().isdigit():
            return True
        
        # Filter single characters
        if len(name.strip()) == 1:
            return True
        
        # Filter common stopwords
        low_quality_terms = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'a', 'an', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were',
            'null', 'none', 'empty', 'unknown', 'undefined', 'n/a', 'na', 'tbd', 'tbc'
        }
        
        if name.strip().lower() in low_quality_terms:
            return True
        
        # Filter pure symbols
        if re.match(r'^[^\w]*$', name.strip()):
            return True
        
        return False
    
    def parse_simple_format(self, output: str) -> Dict[str, Any]:
        """
        Parse simple format output: "ENTITIES: xxx, CONCEPTS: xxx"
        
        Args:
            output: LLM output string
            
        Returns:
            Dictionary with entities and concepts lists
        """
        entities = []
        concepts = []
        
        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            if line.startswith('ENTITIES:') or line.startswith('实体:'):
                # Extract entities
                entity_text = line.split(':', 1)[1].strip()
                if entity_text and entity_text != '':
                    entity_names = [name.strip() for name in entity_text.split(',') if name.strip()]
                    entities = [
                        {"name": self.normalize_concept_name(name)}
                        for name in entity_names
                        if not self.is_low_quality_concept(name)
                    ]
            
            elif line.startswith('CONCEPTS:') or line.startswith('概念:'):
                # Extract concepts
                concept_text = line.split(':', 1)[1].strip()
                if concept_text and concept_text != '':
                    concept_names = [name.strip() for name in concept_text.split(',') if name.strip()]
                    concepts = [
                        {"name": self.normalize_concept_name(name)}
                        for name in concept_names
                        if not self.is_low_quality_concept(name)
                    ]
        
        return {
            "named_entities": entities,
            "document_level_concepts": concepts
        }
    
    def parse_batch_format(self, output: str, expected_count: int) -> List[Dict[str, Any]]:
        """
        Parse batch format output with multiple passages.
        
        Args:
            output: LLM output string
            expected_count: Expected number of passages
            
        Returns:
            List of concept dictionaries
        """
        results = []
        current_entities = []
        current_concepts = []
        
        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            # Detect passage boundaries
            if (line.startswith('=== Passage') or
                line.startswith('PASSAGE_') or
                line.startswith('段落_') or
                (line.startswith('Passage') and ':' in line and len(line.split(':')[0].strip()) < 20)):
                
                # Save previous passage data
                if current_entities or current_concepts:
                    results.append({
                        "named_entities": current_entities,
                        "document_level_concepts": current_concepts
                    })
                
                # Reset for new passage
                current_entities = []
                current_concepts = []
            
            elif line.startswith('ENTITIES:') or line.startswith('实体:'):
                entity_text = line.split(':', 1)[1].strip()
                if entity_text and entity_text != '':
                    entity_names = [name.strip() for name in entity_text.split(',') if name.strip()]
                    current_entities = [
                        {"name": self.normalize_concept_name(name)}
                        for name in entity_names
                        if not self.is_low_quality_concept(name)
                    ]
            
            elif line.startswith('CONCEPTS:') or line.startswith('概念:'):
                concept_text = line.split(':', 1)[1].strip()
                if concept_text and concept_text != '':
                    concept_names = [name.strip() for name in concept_text.split(',') if name.strip()]
                    current_concepts = [
                        {"name": self.normalize_concept_name(name)}
                        for name in concept_names
                        if not self.is_low_quality_concept(name)
                    ]
        
        # Save last passage
        if current_entities or current_concepts:
            results.append({
                "named_entities": current_entities,
                "document_level_concepts": current_concepts
            })
        
        # Handle case where no passage boundaries were found
        if len(results) == 0 and expected_count > 0:
            all_entities = []
            all_concepts = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('ENTITIES:') or line.startswith('实体:'):
                    entity_text = line.split(':', 1)[1].strip()
                    if entity_text:
                        entity_names = [name.strip() for name in entity_text.split(',') if name.strip()]
                        all_entities.extend([
                            {"name": self.normalize_concept_name(name)}
                            for name in entity_names
                            if not self.is_low_quality_concept(name)
                        ])
                elif line.startswith('CONCEPTS:') or line.startswith('概念:'):
                    concept_text = line.split(':', 1)[1].strip()
                    if concept_text:
                        concept_names = [name.strip() for name in concept_text.split(',') if name.strip()]
                        all_concepts.extend([
                            {"name": self.normalize_concept_name(name)}
                            for name in concept_names
                            if not self.is_low_quality_concept(name)
                        ])
            
            if all_entities or all_concepts:
                results.append({
                    "named_entities": all_entities,
                    "document_level_concepts": all_concepts
                })
        
        # Ensure correct result count
        while len(results) < expected_count:
            results.append({"named_entities": [], "document_level_concepts": []})
        
        return results[:expected_count]


class ConceptExtractor:
    """
    Main concept extractor class.
    Extracts named entities and document-level concepts from text passages.
    """
    
    def __init__(
        self,
        config: ConfigLoader,
        llm_client: LLMClient,
        output_dir: str = "output"
    ):
        """
        Initialize concept extractor.
        
        Args:
            config: Configuration loader
            llm_client: LLM client for extraction
            output_dir: Output directory for results
        """
        self.config = config
        self.llm_client = llm_client
        self.output_dir = Path(output_dir)
        
        # Initialize data processor
        self.data_processor = DataProcessor(
            chunk_size=config.extraction.text_chunk_size,
            chunk_overlap=config.extraction.chunk_overlap,
            remove_spaces=config.extraction.remove_doc_spaces
        )
        
        # Initialize output parser
        self.parser = ConceptOutputParser(
            normalize_names=config.extraction.normalize_concept_names,
            filter_low_quality=config.extraction.filter_low_quality
        )
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "concepts").mkdir(exist_ok=True)
        (self.output_dir / "concept_csv").mkdir(exist_ok=True)
        (self.output_dir / "usage").mkdir(exist_ok=True)
    
    def load_dataset(
        self,
        dataset_name: str,
        data_path: str,
        max_samples: int = -1
    ):
        """
        Load benchmark dataset.
        
        Args:
            dataset_name: Name of dataset
            data_path: Path to benchmark data directory
            max_samples: Maximum samples to load (-1 = all)
            
        Returns:
            Loaded dataset
        """
        return self.data_processor.load_benchmark_dataset(
            dataset_name,
            data_path,
            max_samples
        )
    
    def create_extraction_messages(
        self,
        text_batch: List[str],
        system_message: str,
        prompt_template: str
    ) -> List[List[Dict]]:
        """
        Create messages for concept extraction.
        
        Args:
            text_batch: Batch of texts
            system_message: System message
            prompt_template: Prompt template
            
        Returns:
            List of message lists
        """
        messages = []
        
        for text in text_batch:
            user_msg = f"{prompt_template}\n\n{text}"
            
            message = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_msg}
            ]
            messages.append(message)
        
        return messages
    
    def create_batch_extraction_message(
        self,
        text_batch: List[str],
        system_message: str,
        prompt_template: str
    ) -> List[Dict]:
        """
        Create a single message for batch extraction.
        
        Args:
            text_batch: Batch of texts
            system_message: System message
            prompt_template: Prompt template
            
        Returns:
            Single message list
        """
        passages_text = ""
        for i, text in enumerate(text_batch, 1):
            passages_text += f"\n=== Passage {i} ===\n{text}\n"
        
        user_msg = f"{prompt_template}\n{passages_text}"
        
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_msg}
        ]
    
    def extract_concepts_batch(
        self,
        text_batch: List[str],
        batch_size: int = 5
    ) -> List[Dict]:
        """
        Extract concepts using optimized batching.
        
        Args:
            text_batch: Batch of texts
            batch_size: Documents per API call
            
        Returns:
            List of extraction results
        """
        results = []
        
        # Load prompts
        prompts = self.llm_client.prompts["concept_extraction"]
        system_msg = prompts["system_message"]
        passage_prompt = prompts["passage_prompt"]
        
        # Create batched messages
        messages = []
        batch_info = []
        
        for i in range(0, len(text_batch), batch_size):
            current_batch = text_batch[i:i+batch_size]
            message = self.create_batch_extraction_message(
                current_batch,
                system_msg,
                passage_prompt
            )
            messages.append(message)
            batch_info.append(current_batch)
        
        try:
            # Process all batches in parallel
            outputs = self.llm_client.generate_response(
                messages,
                max_new_tokens=self.config.extraction.max_new_tokens * batch_size,
                temperature=self.config.extraction.temperature,
                return_text_only=not self.config.extraction.record_usage,
                max_workers=self.config.extraction.max_workers
            )
            
            # Parse outputs
            for batch_idx, (output, current_batch) in enumerate(zip(outputs, batch_info)):
                if self.config.extraction.record_usage:
                    text_output, usage = output
                else:
                    text_output = output
                    usage = None
                
                # Parse batch output
                batch_results = self.parser.parse_batch_format(text_output, len(current_batch))
                
                # Add results
                for j, parsed_data in enumerate(batch_results):
                    result = {
                        'named_entities': parsed_data.get('named_entities', []),
                        'document_level_concepts': parsed_data.get('document_level_concepts', []),
                        'raw_output': text_output
                    }
                    
                    if usage:
                        # Average usage across batch
                        avg_usage = {k: v/len(current_batch) for k, v in usage.items() if isinstance(v, (int, float))}
                        result['usage'] = avg_usage
                    
                    results.append(result)
        
        except Exception as e:
            print(f"Error in batch concept extraction: {e}")
            # Add empty results on failure
            for _ in text_batch:
                results.append({"named_entities": [], "document_level_concepts": []})
        
        return results
    
    def run_extraction(
        self,
        dataset_name: str,
        data_path: str,
        max_samples: int = -1
    ) -> str:
        """
        Run full concept extraction pipeline.
        
        Args:
            dataset_name: Name of dataset
            data_path: Path to benchmark data directory
            max_samples: Maximum samples to process (-1 = all)
            
        Returns:
            Path to output concept file
        """
        print("=" * 80)
        print("Starting concept extraction...")
        print(f"Dataset: {dataset_name}")
        print(f"Max samples: {max_samples if max_samples > 0 else 'all'}")
        print("=" * 80)
        
        # Load dataset
        dataset = self.load_dataset(dataset_name, data_path, max_samples)
        processed_data = self.data_processor.prepare_dataset(dataset)
        
        print(f"Processing {len(processed_data)} text chunks")
        
        # Create output file
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_file = self.output_dir / "concepts" / f"concepts_{dataset_name}_{timestamp}.jsonl"
        
        # Extract concepts in batches
        batch_size = self.config.extraction.batch_size
        
        if self.config.extraction.optimized_batching:
            print(f"Using optimized batching: {self.config.extraction.optimized_batch_size} documents per API call")
        
        with open(output_file, "w", encoding="utf-8") as f:
            for i in tqdm(range(0, len(processed_data), batch_size), desc="Extracting concepts"):
                batch = processed_data[i:i + batch_size]
                
                # Prepare text batch with titles
                text_batch = []
                for item in batch:
                    title = item['metadata'].get('title', '').strip()
                    text = item['text']
                    if title:
                        combined_text = f"Title: {title}\n\n{text}"
                    else:
                        combined_text = text
                    text_batch.append(combined_text)
                
                # Extract concepts
                if self.config.extraction.optimized_batching:
                    concept_results = self.extract_concepts_batch(
                        text_batch,
                        self.config.extraction.optimized_batch_size
                    )
                else:
                    # Fallback to simple batching
                    concept_results = self.extract_concepts_batch(text_batch, 1)
                
                # Write results
                for j in range(min(len(concept_results), len(batch))):
                    result = concept_results[j]
                    batch_item = batch[j]
                    
                    output_item = {
                        'chunk_id': batch_item['chunk_id'],
                        'metadata': batch_item['metadata'],
                        'text': batch_item['text'],
                        'extracted_concepts': result
                    }
                    
                    f.write(json.dumps(output_item, ensure_ascii=False) + "\n")
        
        print(f"Extraction completed!")
        print(f"Output file: {output_file}")
        
        return str(output_file)
    
    def create_csv_files(self, concept_file: str) -> Tuple[str, str]:
        """
        Create CSV files from concept extraction results.
        
        Args:
            concept_file: Path to concept JSONL file
            
        Returns:
            Tuple of (concepts_csv_path, passages_csv_path)
        """
        print("Creating CSV files...")
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        concepts_csv = self.output_dir / "concept_csv" / f"concepts_{timestamp}.csv"
        passages_csv = self.output_dir / "concept_csv" / f"passages_{timestamp}.csv"
        
        all_concepts = []
        all_passages = []
        concept_id_counter = 0
        passage_id_counter = 0
        
        with open(concept_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    extracted = data.get('extracted_concepts', {})
                    chunk_id = data.get('chunk_id', '')
                    passage_text = data.get('text', '')
                    metadata = data.get('metadata', {})
                    
                    # Handle merged source information
                    source_docs = metadata.get('source_docs', [])
                    source_files = metadata.get('source_files', [])
                    
                    source_doc = ','.join(list(set(source_docs))) if source_docs else metadata.get('dataset', '')
                    merged_chunk_id = ','.join(list(set(source_files))) if source_files else chunk_id
                    
                    # Create passage record
                    passage_row = {
                        'passage_id': f"passage_{passage_id_counter}",
                        'chunk_id': merged_chunk_id,
                        'text': passage_text,
                        'source_doc': source_doc,
                        'title': metadata.get('title', ''),
                        'dataset': metadata.get('dataset', ''),
                        'word_count': len(passage_text.split()),
                        'text_hash': metadata.get('text_hash', '')
                    }
                    all_passages.append(passage_row)
                    current_passage_id = passage_row['passage_id']
                    passage_id_counter += 1
                    
                    # Process entities
                    for concept in extracted.get('named_entities', []):
                        if isinstance(concept, dict) and 'name' in concept:
                            concept_row = {
                                'concept_id': f"concept_{concept_id_counter}",
                                'name': concept.get('name', ''),
                                'type': 'entity',
                                'abstraction_level': 'specific',
                                'source_doc': source_doc,
                                'passage_id': current_passage_id,
                                'chunk_id': merged_chunk_id
                            }
                            all_concepts.append(concept_row)
                            concept_id_counter += 1
                    
                    # Process document-level concepts
                    for concept in extracted.get('document_level_concepts', []):
                        if isinstance(concept, dict) and 'name' in concept:
                            concept_row = {
                                'concept_id': f"concept_{concept_id_counter}",
                                'name': concept.get('name', ''),
                                'type': 'document',
                                'abstraction_level': 'abstract',
                                'source_doc': source_doc,
                                'passage_id': current_passage_id,
                                'chunk_id': merged_chunk_id
                            }
                            all_concepts.append(concept_row)
                            concept_id_counter += 1
                
                except Exception as e:
                    print(f"Error processing line: {e}")
                    continue
        
        # Write concepts CSV
        if all_concepts:
            concept_fieldnames = ['concept_id', 'name', 'type', 'abstraction_level',
                                  'source_doc', 'passage_id', 'chunk_id']
            with open(concepts_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=concept_fieldnames)
                writer.writeheader()
                writer.writerows(all_concepts)
        
        # Write passages CSV
        if all_passages:
            passage_fieldnames = ['passage_id', 'chunk_id', 'text', 'source_doc',
                                  'title', 'dataset', 'word_count', 'text_hash']
            with open(passages_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=passage_fieldnames)
                writer.writeheader()
                writer.writerows(all_passages)
        
        print(f"Created concepts CSV: {concepts_csv} ({len(all_concepts)} concepts)")
        print(f"Created passages CSV: {passages_csv} ({len(all_passages)} passages)")
        
        return str(concepts_csv), str(passages_csv)
    
    def aggregate_usage_stats(self, concept_file: str) -> Tuple[str, Dict]:
        """
        Aggregate token usage statistics.
        
        Args:
            concept_file: Path to concept JSONL file
            
        Returns:
            Tuple of (usage_file_path, usage_summary_dict)
        """
        print("Aggregating token usage statistics...")
        
        totals = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'time': 0.0
        }
        num_calls = 0
        
        with open(concept_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    extracted = obj.get('extracted_concepts', {})
                    usage = extracted.get('usage')
                    
                    if isinstance(usage, dict):
                        prompt = usage.get('prompt_tokens', usage.get('input_tokens', 0)) or 0
                        completion = usage.get('completion_tokens', usage.get('output_tokens', 0)) or 0
                        total = usage.get('total_tokens', 0) or (prompt + completion)
                        time_cost = usage.get('time', 0.0) or 0.0
                        
                        totals['prompt_tokens'] += float(prompt)
                        totals['completion_tokens'] += float(completion)
                        totals['total_tokens'] += float(total)
                        totals['time'] += float(time_cost)
                        num_calls += 1
                
                except Exception:
                    continue
        
        usage_summary = {
            'model': self.llm_client.model_name,
            'num_calls': num_calls,
            'totals': totals,
            'avg_per_call': {
                'prompt_tokens': totals['prompt_tokens'] / num_calls if num_calls else 0,
                'completion_tokens': totals['completion_tokens'] / num_calls if num_calls else 0,
                'total_tokens': totals['total_tokens'] / num_calls if num_calls else 0,
                'time': totals['time'] / num_calls if num_calls else 0.0
            },
            'created_at': datetime.now().isoformat(),
            'concept_output_file': concept_file
        }
        
        # Save usage file
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        usage_file = self.output_dir / "usage" / f"token_usage_{timestamp}.json"
        
        with open(usage_file, 'w', encoding='utf-8') as f:
            json.dump(usage_summary, f, ensure_ascii=False, indent=2)
        
        print(f"Token usage saved: {usage_file}")
        print(f"Total calls: {num_calls}, Total tokens: {int(totals['total_tokens'])}")
        
        return str(usage_file), usage_summary

