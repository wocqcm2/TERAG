"""
Data processing utilities for concept extraction.
Handles text chunking, data loading, and preprocessing.
"""

import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from datasets import load_dataset, Dataset


# Token approximation constant (1 token ≈ 3.5 characters)
CHAR_TO_TOKEN_RATIO = 3.5


class TextChunker:
    """
    Text chunker using character-based token approximation.
    Splits long texts into manageable chunks with overlap.
    """
    
    def __init__(self, chunk_size: int = 1024, overlap: int = 100):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Maximum tokens per chunk
            overlap: Overlap between chunks (in tokens)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.char_ratio = CHAR_TO_TOKEN_RATIO
        
        print(f"Initialized TextChunker: chunk_size={chunk_size} tokens, overlap={overlap} tokens")
        print(f"Using character approximation: 1 token ≈ {self.char_ratio} characters")
    
    def calculate_max_chars(self) -> int:
        """Calculate maximum characters per chunk."""
        return int(self.chunk_size * self.char_ratio)
    
    def calculate_overlap_chars(self) -> int:
        """Calculate overlap in characters."""
        return int(self.overlap * self.char_ratio)
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count from character count."""
        return int(len(text) / self.char_ratio)
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        max_chars = self.calculate_max_chars()
        overlap_chars = self.calculate_overlap_chars()
        
        # If text is short enough, return as-is
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chars
            
            # Try to split at sentence boundary
            if end < len(text):
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + max_chars // 2:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Calculate next start position with overlap
            if overlap_chars > 0:
                start = max(end - overlap_chars, start + 10)
            else:
                start = end
            
            if start >= len(text):
                break
        
        return chunks


class DataProcessor:
    """
    Data processor for concept extraction.
    Handles loading and preprocessing of benchmark datasets.
    """
    
    def __init__(
        self,
        chunk_size: int = 4096,
        chunk_overlap: int = 150,
        remove_spaces: bool = True
    ):
        """
        Initialize data processor.
        
        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Overlap between chunks (in tokens)
            remove_spaces: Whether to normalize whitespace
        """
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.remove_spaces = remove_spaces
    
    def load_benchmark_dataset(
        self,
        dataset_name: str,
        data_path: str,
        max_samples: int = -1
    ) -> Dataset:
        """
        Load benchmark dataset from JSON file.
        
        Args:
            dataset_name: Name of dataset (hotpotqa, 2wikimultihopqa, musique)
            data_path: Path to benchmark data directory
            max_samples: Maximum samples to load (-1 = all)
            
        Returns:
            Dataset object
        """
        # File mapping
        file_mapping = {
            "musique": "musique.json",
            "hotpotqa": "hotpotqa.json",
            "2wikimultihopqa": "2wikimultihopqa.json"
        }
        
        if dataset_name not in file_mapping:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        data_file = Path(data_path) / file_mapping[dataset_name]
        
        if not data_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_file}")
        
        print(f"Loading benchmark data: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Limit samples if specified
        if max_samples > 0 and len(data) > max_samples:
            data = data[:max_samples]
            print(f"Using first {max_samples} samples")
        else:
            print(f"Using all {len(data)} samples")
        
        # Convert to TERAG format
        converted_data = self._convert_benchmark_format(data, dataset_name)
        
        # Create Dataset object
        dataset = Dataset.from_list(converted_data)
        return dataset
    
    def _convert_benchmark_format(
        self,
        data: List[Dict],
        dataset_name: str
    ) -> List[Dict]:
        """
        Convert benchmark data to unified format.
        
        Args:
            data: Raw benchmark data
            dataset_name: Name of dataset
            
        Returns:
            List of processed data items
        """
        converted = []
        item_id = 0
        
        for sample in data:
            # Extract text segments based on dataset format
            text_segments = self._extract_text_segments(sample, dataset_name)
            
            for text, title in text_segments:
                converted_item = {
                    "text": text,
                    "metadata": {
                        "lang": "en",
                        "source": f"{dataset_name}_benchmark",
                        "title": title,
                        "sample_id": sample.get('_id', ''),
                        "dataset": dataset_name,
                        "file_id": f"{dataset_name}_{item_id}"
                    }
                }
                converted.append(converted_item)
                item_id += 1
        
        print(f"Converted {len(converted)} text passages")
        return converted
    
    def _extract_text_segments(
        self,
        sample: Dict,
        dataset_name: str
    ) -> List[Tuple[str, str]]:
        """
        Extract text segments from sample.
        
        Args:
            sample: Single data sample
            dataset_name: Name of dataset
            
        Returns:
            List of (text, title) tuples
        """
        if dataset_name == "musique":
            # MuSiQue format: paragraphs field
            paragraphs = sample.get('paragraphs', [])
            return [
                (para['paragraph_text'], para['title'])
                for para in paragraphs
            ]
        elif dataset_name in ["hotpotqa", "2wikimultihopqa"]:
            # HotpotQA/2WikiMultiHopQA format: context field
            context_data = sample.get('context', [])
            segments = []
            for title, paragraphs in context_data:
                full_text = ' '.join(paragraphs) if isinstance(paragraphs, list) else paragraphs
                segments.append((full_text, title))
            return segments
        else:
            return []
    
    def prepare_dataset(self, dataset: Dataset) -> List[Dict[str, Any]]:
        """
        Prepare dataset for concept extraction.
        Performs text chunking and deduplication.
        
        Args:
            dataset: Input dataset
            
        Returns:
            List of processed samples
        """
        processed_samples = []
        seen_texts = set()
        text_to_metadata = {}
        
        print("Preprocessing and deduplicating text...")
        
        for item in tqdm(dataset, desc="Processing dataset"):
            text = item.get('text', '')
            metadata = item.get('metadata', {})
            
            # Normalize whitespace if enabled
            if self.remove_spaces:
                text = re.sub(r'\s+', ' ', text).strip()
            
            # Skip empty or very short texts
            if not text or len(text.strip()) < 10:
                continue
            
            # Chunk long texts
            text_chunks = self.chunker.chunk_text(text)
            
            for i, chunk in enumerate(text_chunks):
                chunk = chunk.strip()
                if not chunk or len(chunk) < 10:
                    continue
                
                # Calculate text hash for deduplication
                chunk_hash = hashlib.md5(chunk.encode('utf-8')).hexdigest()
                
                if chunk_hash in seen_texts:
                    # Merge metadata for duplicate texts
                    existing_metadata = text_to_metadata[chunk_hash]
                    existing_metadata['source_files'].append(metadata.get('file_id', 'unknown'))
                    existing_metadata['source_docs'].append(metadata.get('dataset', 'unknown'))
                    continue
                
                # Mark as seen
                seen_texts.add(chunk_hash)
                
                # Create metadata for new chunk
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_id'] = i
                chunk_metadata['total_chunks'] = len(text_chunks)
                chunk_metadata['original_length'] = len(text)
                chunk_metadata['text_hash'] = chunk_hash
                chunk_metadata['source_files'] = [metadata.get('file_id', 'unknown')]
                chunk_metadata['source_docs'] = [metadata.get('dataset', 'unknown')]
                
                text_to_metadata[chunk_hash] = chunk_metadata
                
                processed_samples.append({
                    'text': chunk,
                    'metadata': chunk_metadata,
                    'chunk_id': f"{metadata.get('file_id', 'unknown')}_{i}",
                    'text_hash': chunk_hash
                })
        
        # Update metadata with merged source information
        for sample in processed_samples:
            chunk_hash = sample['text_hash']
            if chunk_hash in text_to_metadata:
                sample['metadata'] = text_to_metadata[chunk_hash]
        
        original_count = len(dataset)
        final_count = len(processed_samples)
        dedup_rate = ((original_count - final_count) / original_count * 100) if original_count > 0 else 0
        
        print(f"Text deduplication complete!")
        print(f"Original text chunks: {original_count}")
        print(f"After deduplication: {final_count}")
        print(f"Deduplication rate: {dedup_rate:.1f}%")
        
        return processed_samples

