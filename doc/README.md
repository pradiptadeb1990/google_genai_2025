# Narrative Style Transferer Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Implementation Details](#implementation-details)
5. [Models and Technologies](#models-and-technologies)
6. [Future Improvements](#future-improvements)

## Project Overview

The Narrative Style Transferer is a sophisticated system that transforms story summaries into specific genre styles using multiple GenAI capabilities. The system combines RAG (Retrieval Augmented Generation), vector search, few-shot learning, structured output, and function calling to achieve high-quality style transformations while maintaining narrative coherence.

## System Architecture

```
[User Input: Story Summary]
           ↓
[Style Selection]
           ↓
┌─────────────────────┐
│  Vector Database    │←──[Style Corpus]
│  (ChromaDB)        │    (Processed & Embedded)
└─────────────────┬──┘
           ↓      │
[Style Examples]  │    ┌─────────────────┐
           ↓      └───→│  Few-Shot       │
[Retrieved Examples]   │  Examples       │
           ↓          └─────────────────┘
┌─────────────────────┐
│  Prompt Assembly    │
│  with Examples      │
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│  Gemini Pro LLM    │
│  Generation        │
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│  Style Analysis    │
│  (Function Calling) │
└─────────┬───────────┘
          ↓
[Styled Narrative Output]
```

## Core Components

### 1. Data Processing (`data_processing.py`)
- Handles text processing and embedding generation
- Manages few-shot examples
- Components:
  - `StyleCorpusProcessor`: Processes and embeds style examples
  - `FewShotExampleManager`: Manages curated examples for each genre

### 2. Vector Database (`vector_db.py`)
- Implements vector storage and retrieval using ChromaDB
- Enables semantic search for style examples
- Features:
  - Efficient embedding storage
  - Genre-specific filtering
  - Similarity search

### 3. Retriever (`retriever.py`)
- Combines vector search and few-shot examples
- Implements hybrid retrieval strategy
- Components:
  - Style example retrieval
  - Example formatting for prompts
  - Style analysis integration

### 4. Generator (`generator.py`)
- Handles LLM interaction and generation
- Implements structured output and function calling
- Features:
  - JSON-structured story generation
  - Dynamic prompt assembly
  - Temperature control

### 5. Style Analysis (`functions/style_analyzer.py`)
- Analyzes generated text for style adherence
- Provides detailed style metrics
- Features:
  - Vocabulary analysis
  - Sentence structure analysis
  - Genre-specific marker detection

## Implementation Details

### RAG Implementation
The RAG system is implemented through a hybrid approach:

1. **Vector Search**:
   - Uses sentence-transformers for embedding generation
   - Stores embeddings in ChromaDB for efficient retrieval
   - Retrieves semantically similar style examples

2. **Example Selection**:
   ```python
   # Example retrieval process
   retrieved_examples = vector_db.query(
       query_text=summary,
       n_results=3,
       genre_filter=target_genre
   )
   ```

3. **Prompt Augmentation**:
   - Combines retrieved examples with few-shot examples
   - Structures examples for optimal context

### Few-Shot Learning
The system uses a curated set of examples for each genre:

1. **Example Structure**:
   ```json
   {
     "original": "Original story summary",
     "styled": "Styled version in target genre"
   }
   ```

2. **Dynamic Selection**:
   - Chooses most relevant examples based on context
   - Balances example length and variety

### Structured Output
Implements JSON-structured generation:

1. **Story Schema**:
   ```json
   {
     "title": "Story title",
     "genre": "Target genre",
     "author_note": "Style explanation",
     "sections": [
       {
         "title": "Section title",
         "content": "Section content"
       }
     ],
     "style_elements": [
       {
         "element_type": "vocabulary",
         "description": "Description",
         "examples": ["example1", "example2"]
       }
     ]
   }
   ```

### Function Calling
Implements dynamic style analysis:

1. **Style Analysis Function**:
   - Vocabulary analysis
   - Sentence structure analysis
   - Genre marker detection

2. **Parameter Tuning**:
   - Dynamic temperature adjustment
   - Style intensity control

## Models and Technologies

### 1. Language Models
- **Gemini Pro**
  - Primary LLM for generation
  - Advantages:
    - Strong structured output capabilities
    - Good function calling support
    - High-quality text generation
    - Context window suitable for story generation

### 2. Embedding Models
- **all-MiniLM-L6-v2** (Sentence Transformers)
  - Used for style example embedding
  - Advantages:
    - Good balance of performance and efficiency
    - Well-suited for semantic similarity
    - Small model size for quick inference

### 3. Vector Database
- **ChromaDB**
  - Stores and retrieves style examples
  - Advantages:
    - Efficient similarity search
    - Good integration with Python
    - Support for metadata filtering

### 4. Supporting Libraries
- **LangChain**: For RAG implementation
- **Pydantic**: For data validation
- **FAISS**: For efficient vector search
- **Sentence Transformers**: For embedding generation

## Future Improvements

### 1. Enhanced Style Analysis
- Implement more sophisticated NLP techniques
- Add support for:
  - Sentiment analysis
  - Narrative arc detection
  - Character voice consistency

### 2. Advanced RAG
- Implement hybrid search strategies
- Add cross-encoder reranking
- Develop dynamic example weighting

### 3. Model Improvements
- Experiment with different embedding models
- Add support for multiple LLMs
- Implement model fallbacks

### 4. Feature Additions
- Add support for:
  - Multiple languages
  - Custom genre definitions
  - Style mixing
  - Interactive style adjustment

### 5. Performance Optimization
- Implement caching strategies
- Add batch processing
- Optimize vector search

### 6. User Experience
- Add progress tracking
- Implement style preview
- Add explanation features

### 7. Quality Assurance
- Add automated testing
- Implement style consistency checks
- Add content safety filters
