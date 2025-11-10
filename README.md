# TERAG: Token-Efficient Graph-Based  Retrieval-Augmented Generation

A lightweight and token-efficient framework for knowledge graph construction and retrieval-augmented generation (RAG) designed for multi-hop question answering. TERAG reduces token consumption during graph construction to only **3-11% of existing methods**.

This project builds upon and extends the HippoRAG framework with improved retrieval mechanisms and comprehensive evaluation tools.

*   **Paper:** [Read the paper](https://arxiv.org/abs/2509.18667)
*   **Datasets:** We use 1,000-sample subsets of HotpotQA, 2WikiMultiHopQA, and MuSiQue extracted by [AutoSchemaKG](https://github.com/HKUST-KnowComp/AutoSchemaKG)

## TERAG Overview

TERAG introduces a **token-efficient** end-to-end pipeline for multi-hop question answering over knowledge graphs:

### Key Features

1. **Token-Efficient Concept Extraction**: Reduces token consumption to **3-11% of state-of-the-art methods** through optimized batching, intelligent deduplication, and efficient prompting strategies
2. **Lightweight Knowledge Graph Construction**: Builds structured knowledge graphs with concept and passage nodes using minimal LLM calls, connected through co-occurrence relationships
3. **Enhanced Retrieval**: Implements both original and enhanced versions of HippoRAG, incorporating named entity recognition, personalized PageRank, and frequency-based re-ranking
4. **Comprehensive Evaluation**: Provides detailed metrics including Exact Match (EM), F1 score, and Recall@K for thorough performance analysis

The framework achieves competitive performance across multiple multi-hop QA benchmarks while dramatically reducing computational costs, making it practical for large-scale deployments.

## Project Structure

```
terag/
├── config/                   # Configuration files
│   ├── config.yaml          # Main configuration
│   └── prompts.yaml         # LLM prompt templates
├── terag/                   # Main package directory
│   ├── extraction/          # Concept extraction modules
│   │   ├── concept_extractor.py
│   │   └── data_processor.py
│   ├── graph/               # Knowledge graph construction
│   │   └── graph_builder.py
│   ├── retrieval/           # Retrieval components
│   │   ├── hipporag_original.py
│   │   └── hipporag_enhanced.py
│   ├── benchmark/           # Evaluation modules
│   │   └── rag_evaluator.py
│   └── utils/               # Utility functions
│       ├── config_loader.py
│       └── llm_client.py
├── dataset/                 # Benchmark datasets
│   ├── hotpotqa.json
│   ├── 2wikimultihopqa.json
│   └── musique.json
├── output/                  # Pipeline output directory (generated)
│   └── {dataset_name}_{num_samples}/
│       ├── extraction/      # Step 1 outputs
│       │   ├── concepts/    # Raw extraction results (.jsonl)
│       │   ├── concept_csv/ # Processed concept & passage nodes (.csv)
│       │   └── usage/       # Token usage statistics (.json)
│       ├── graph/           # Step 2 outputs
│       │   ├── *.graphml    # Knowledge graph files
│       │   └── *.stats.json # Graph statistics
│       └── evaluation/      # Step 3 outputs
│           └── *.json       # Evaluation results
├── scripts/                 # Example scripts
├── pipeline.py              # Main pipeline script
└── requirements.txt         # Package dependencies
```

The project is organized into several key components:
- `terag/`: Core package containing extraction, graph building, retrieval, and evaluation modules
- `config/`: Configuration files for API settings, model parameters, and prompts
- `dataset/`: Storage for benchmark datasets
- `output/`: Generated outputs including extracted concepts, knowledge graphs, token usage statistics, and evaluation results
- `pipeline.py`: Unified pipeline for end-to-end processing

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for embedding generation)
- API access to OpenAI-compatible LLM services (e.g., DeepInfra, OpenAI)

### Setup Conda Environment

```bash
# Create a new conda environment
conda create -n terag python=3.10
conda activate terag

# Clone the repository
git clone <your-repository-url>
cd terag

# Install required packages
pip install -r requirements.txt
```

> **Note:** The project is designed to run directly from the repository root directory without requiring package installation. The `pipeline.py` script automatically handles import paths.

### Configure API Keys

Edit `config/config.yaml` to set your LLM API credentials:

```yaml
api:
  provider: "deepinfra"  # Options: "deepinfra" or "openai"
  deepinfra:
    api_key: "your-deepinfra-api-key"
    base_url: "https://api.deepinfra.com/v1/openai"
  openai:
    api_key: "your-openai-api-key"
    base_url: "https://api.openai.com/v1"
```

### Recommended Models

**For concept extraction:**
- **DeepInfra**: `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`

**For answer generation:**
- **DeepInfra**: `meta-llama/Llama-3.3-70B-Instruct`

**For embeddings:**
- **Default**: `all-MiniLM-L6-v2` 

## Building New Knowledge Graphs and Implementing RAG

### Full Pipeline Example

The complete pipeline includes concept extraction, graph construction, and RAG evaluation:

```bash
# Run the full pipeline on HotpotQA (1000 samples)
python pipeline.py \
  --dataset hotpotqa \
  --max_samples 1000 \
  --output_dir output/hotpotqa_1000 \
  --retrievers enhanced
```

### Step-by-Step Pipeline

For more control, you can run each step individually:

#### Step 1: Concept Extraction

Extract named entities and document-level concepts from your dataset:

```bash
python pipeline.py \
  --dataset hotpotqa \
  --step 1 \
  --max_samples 1000 \
  --output_dir output/hotpotqa_1000
```

**Output:**
- `output/hotpotqa_1000/extraction/concepts/*.jsonl`: Raw extraction results
- `output/hotpotqa_1000/extraction/concept_csv/concepts_*.csv`: Concept nodes
- `output/hotpotqa_1000/extraction/concept_csv/passages_*.csv`: Passage nodes
- `output/hotpotqa_1000/extraction/usage/token_usage_*.json`: Token consumption statistics (tracks API calls, prompt/completion tokens, and processing time)

#### Step 2: Knowledge Graph Construction

Build a knowledge graph from extracted concepts:

```bash
python pipeline.py \
  --dataset hotpotqa \
  --step 2 \
  --output_dir output/hotpotqa_1000
```

**Output:**
- `output/hotpotqa_1000/graph/knowledge_graph_*.graphml`: NetworkX-compatible graph file
- `output/hotpotqa_1000/graph/knowledge_graph_*.stats.json`: Graph statistics

> **Note:** After graph construction, the pipeline automatically displays a summary of token usage from Step 1, showing total API calls, prompt/completion tokens, and processing time.

#### Step 3: RAG Evaluation

Evaluate retrieval and generation performance:

```bash
python pipeline.py \
  --dataset hotpotqa \
  --step 3 \
  --eval_samples 1000 \
  --retrievers enhanced \
  --output_dir output/hotpotqa_1000
```

**Output:**
- `output/hotpotqa_1000/benchmark/results_*.json`: Detailed evaluation results

### Using Custom Datasets

To build knowledge graphs from your own data:

1. **Prepare your data** in one of the supported formats (see [Supported Data Formats](#supported-data-formats))
2. **Add to configuration** in `config/config.yaml`:
   ```yaml
   benchmark:
     datasets:
       my_dataset: "dataset/my_dataset.json"
   ```
3. **Run the pipeline**:
   ```bash
   python pipeline.py --dataset my_dataset --max_samples -1 --output_dir output/my_dataset
   ```

## Multi-hop Question Answering Evaluation

TERAG supports evaluation on three major multi-hop QA benchmarks:

### Supported Benchmarks

We evaluate on three major multi-hop QA benchmarks using 1,000-sample subsets extracted by [AutoSchemaKG](https://github.com/HKUST-KnowComp/AutoSchemaKG):

- **HotpotQA**: Wikipedia-based multi-hop reasoning dataset with diverse question types
- **2WikiMultiHopQA**: Multi-hop questions requiring complex reasoning across Wikipedia articles
- **MuSiQue**: Answerable and unanswerable multi-hop questions with reasoning decomposition

### Running Evaluations

#### Evaluate on a Single Dataset

```bash
# Evaluate HotpotQA with Enhanced HippoRAG
python pipeline.py \
  --dataset hotpotqa \
  --max_samples 1000 \
  --output_dir output/hotpotqa_1000 \
  --retrievers enhanced
```

#### Compare Original vs Enhanced Retrievers

```bash
# Test both retriever versions
python pipeline.py \
  --dataset 2wikimultihopqa \
  --max_samples 1000 \
  --output_dir output/2wiki_comparison \
  --retrievers both
```

#### Reuse Existing Knowledge Graphs

To save time, reuse previously constructed graphs for repeated experiments:

```bash
# Only run evaluation step (requires existing graph)
python pipeline.py \
  --dataset hotpotqa \
  --step 3 \
  --eval_samples 1000 \
  --retrievers enhanced \
  --output_dir output/hotpotqa_1000
```

### Evaluation Metrics

The evaluation provides comprehensive metrics:

- **Exact Match (EM)**: Percentage of predictions that exactly match the ground truth
- **F1 Score**: Token-level F1 score between prediction and ground truth
- **Recall@2**: Percentage of questions where supporting documents appear in top-2 retrieved passages
- **Recall@5**: Percentage of questions where supporting documents appear in top-5 retrieved passages
- **Retrieval Time**: Average time per retrieval operation

### Results Interpretation

Example evaluation output:

```
Evaluation Summary:

HippoRAG_Enhanced:
  EM:        0.5090
  F1:        0.5719
  Recall@2:  0.5557
  Recall@5:  0.6705
```

This indicates:
- 50.9% of answers are completely correct
- 57.2% average token overlap with correct answers
- Supporting documents found in top-2 results 55.6% of the time
- Supporting documents found in top-5 results 67.1% of the time

## Supported Data Formats

TERAG supports two standard multi-hop QA data formats:

### Format 1: HotpotQA / 2WikiMultiHopQA

```json
[
    {
        "_id": "unique_sample_id",
        "question": "What is the question text?",
        "answer": "The answer text",
        "supporting_facts": [
            ["Document Title 1", 0],
            ["Document Title 2", 1]
        ],
        "context": [
            [
                "Document Title 1",
                [
                    "Paragraph 1 text...",
                    "Paragraph 2 text..."
                ]
            ],
            [
                "Document Title 2",
                [
                    "Paragraph 1 text..."
                ]
            ]
        ]
    }
]
```

**Required Fields:**
- `_id`: Unique identifier for the sample
- `question`: The question text
- `answer`: Ground truth answer
- `supporting_facts`: List of [title, paragraph_index] indicating supporting documents
- `context`: List of [title, paragraphs] containing all documents

### Format 2: MuSiQue

```json
[
    {
        "_id": "unique_sample_id",
        "question": "What is the question text?",
        "answer": "The answer text",
        "paragraphs": [
            {
                "idx": 0,
                "title": "Document Title 1",
                "paragraph_text": "Full paragraph text...",
                "is_supporting": true
            },
            {
                "idx": 1,
                "title": "Document Title 2",
                "paragraph_text": "Full paragraph text...",
                "is_supporting": false
            }
        ]
    }
]
```

**Required Fields:**
- `_id`: Unique identifier
- `question`: The question text
- `answer`: Ground truth answer
- `paragraphs`: List of paragraph objects with:
  - `title`: Document title
  - `paragraph_text`: Full paragraph content
  - `is_supporting`: Boolean indicating if this is a supporting document

## Adding Custom Datasets

To add your own benchmark dataset:

### Step 1: Convert Your Data

Convert your dataset to one of the supported formats above. Save as JSON file in the `dataset/` directory.

### Step 2: Update Configuration

Add your dataset to `config/config.yaml`:

```yaml
benchmark:
  datasets:
    hotpotqa: "dataset/hotpotqa.json"
    2wikimultihopqa: "dataset/2wikimultihopqa.json"
    musique: "dataset/musique.json"
    my_custom_dataset: "dataset/my_custom_dataset.json"  # Add this line
```

### Step 3: (Optional) Extend Data Processing

If your format differs significantly, modify the data extraction logic:

**For concept extraction** (`terag/extraction/data_processor.py`):

```python
def _extract_text_segments(self, sample: Dict, dataset_name: str):
    if dataset_name == "my_custom_dataset":
        # Your custom extraction logic
        return [(text, title), ...]
    # ... existing code
```

**For evaluation** (`terag/benchmark/rag_evaluator.py`):

```python
def extract_supporting_facts(self, sample: Dict, dataset_name: str):
    if dataset_name == "my_custom_dataset":
        # Your custom supporting facts extraction
        return [list of titles]
    # ... existing code
```

### Step 4: Run Pipeline

```bash
python pipeline.py \
  --dataset my_custom_dataset \
  --max_samples 1000 \
  --output_dir output/my_custom_dataset
```

### Data Validation Script

Use this script to validate your dataset format:

```python
import json

def validate_dataset(file_path, format_type="hotpotqa"):
    """Validate dataset format"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    required_fields = ["_id", "question", "answer"]
    
    if format_type in ["hotpotqa", "2wikimultihopqa"]:
        required_fields += ["context", "supporting_facts"]
    elif format_type == "musique":
        required_fields += ["paragraphs"]
    
    errors = []
    for i, sample in enumerate(data):
        missing = [f for f in required_fields if f not in sample]
        if missing:
            errors.append(f"Sample {i}: missing {missing}")
    
    if errors:
        print(f"Found {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10
            print(f"  - {error}")
    else:
        print(f"✓ Dataset validation passed ({len(data)} samples)")

# Usage
validate_dataset("dataset/my_dataset.json", "hotpotqa")
```

## Advanced Configuration

### Customizing Prompts

Edit `config/prompts.yaml` to modify LLM prompts:

```yaml
# Chain-of-Thought prompting for answer generation
answer_generation:
  system_message: |
    As an advanced reading comprehension assistant, your task is to analyze 
    text passages and corresponding questions meticulously. Your response 
    starts with "Thought: " followed by step-by-step reasoning, and concludes 
    with "Answer: " providing a concise response.
```

### Adjusting Retrieval Parameters

In `config/config.yaml`, tune retrieval performance:

```yaml
retrieval:
  ppr_alpha: 0.55              # PageRank damping factor (higher = more local)
  ppr_topk: 10                 # Number of nodes to retrieve
  damping_factor: 0.85         # Random walk restart probability
  enhanced_freq_weight: 0.3    # Weight for frequency-based re-ranking (Enhanced only)
```

### Controlling Token Usage

Configure batching and chunking to manage API costs:

```yaml
extraction:
  batch_size: 5                # Documents per API call
  text_chunk_size: 4096        # Maximum tokens per chunk
  max_workers: 10              # Parallel extraction workers
```

## Citation

If you use TERAG in your research, please cite:

```bibtex
@misc{xiao2025teragtokenefficientgraphbasedretrievalaugmented,
  title={TERAG: Token-Efficient Graph-Based Retrieval-Augmented Generation}, 
  author={Qiao Xiao and Hong Ting Tsang and Jiaxin Bai},
  year={2025},
  eprint={2509.18667},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2509.18667}
}
```

## Acknowledgments

This project builds upon the [HippoRAG](https://arxiv.org/abs/2405.14831) framework for retrieval mechanisms and uses benchmark datasets from [AutoSchemaKG](https://github.com/HKUST-KnowComp/AutoSchemaKG). We thank the authors of these works for their foundational contributions to the field of knowledge graph construction and retrieval-augmented generation.

## Contact

**Jiaxin Bai**  
Email: jbai@connect.ust.hk

**Xiaoqiao Qiao**  
Email: qx226@cornell.edu

For questions, issues, or collaboration opportunities, please feel free to reach out or open an issue on GitHub.

## License

MIT License - See LICENSE file for details
