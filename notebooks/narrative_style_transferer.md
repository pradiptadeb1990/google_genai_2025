# Narrative Style Transferer: A GenAI Capstone Project

## Introduction

This project demonstrates a sophisticated narrative style transfer system that can transform story summaries into specific genre styles (e.g., turning a sci-fi story into Lovecraftian horror). The system leverages multiple GenAI capabilities to achieve high-quality style transformations while maintaining the core narrative elements.

### GenAI Capabilities Demonstrated

1. **Retrieval Augmented Generation (RAG)**
   - Vector-based retrieval of style examples
   - Dynamic prompt augmentation with relevant examples
   - Hybrid retrieval combining semantic search and curated examples

2. **Vector Search/Vector Database**
   - Efficient storage and retrieval of style examples
   - Semantic similarity search for finding relevant passages
   - Genre-specific indexing and retrieval

3. **Few-Shot Prompting**
   - Curated examples for each genre style
   - Dynamic example selection based on context
   - Demonstration-based learning for style transfer

4. **Structured Output**
   - JSON-structured story generation
   - Consistent narrative organization
   - Detailed style analysis output

5. **Function Calling**
   - Dynamic style analysis
   - Genre classification
   - Parameter tuning based on content

## Setup and Dependencies

```python
# Install required packages
!pip install -q google-generativeai==0.3.0 \
    langchain>=0.1.0 \
    langchain-community>=0.0.10 \
    chromadb>=0.4.18 \
    sentence-transformers>=2.2.2 \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    tqdm>=4.66.0 \
    faiss-cpu>=1.7.4 \
    ipywidgets>=8.0.0

# Import necessary libraries
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import google.generativeai as genai
from IPython.display import display, HTML, Markdown
```

## API Configuration

```python
# Configure Gemini API
# Note: In Kaggle, use Kaggle Secrets to manage your API key
import os
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
```

## 1. RAG Implementation

### Vector Store Setup

```python
from src.vector_store import StyleVectorStore
from src.data_processing import StyleCorpusProcessor

# Initialize vector store (uses LlamaIndex with ChromaDB backend)
vector_store = StyleVectorStore(
    persist_directory="vector_store",
    use_existing=True  # Use existing database if available
)

# Optional: Process and add new documents
corpus_processor = StyleCorpusProcessor()
corpus = corpus_processor.load_corpus("style_corpus")
processed_corpus = corpus_processor.create_embeddings(corpus)

# Add to vector store
for genre, chunks in processed_corpus.items():
    vector_store.add_documents(chunks)
```

The vector store uses LlamaIndex with a ChromaDB backend for persistent storage. This means:
- Documents are automatically persisted to disk
- The database can be reused across sessions
- Embeddings are cached for better performance
- Supports semantic search with filters

### Example Retrieval

```python
# Query for style examples
results = vector_store.query(
    query_text="A detective investigates a mysterious artifact",
    n_results=3,
    genre_filter="lovecraftian"
)

# Display retrieved examples
for result in results:
    print(f"Example (Distance: {result['distance']:.3f}):")
    print(result['text'])
    print("-" * 80)
```

## 2. Few-Shot Prompting

### Example Management

```python
from src.data_processing import FewShotExampleManager

# Initialize example manager
few_shot_manager = FewShotExampleManager("few_shot_examples")

# Get examples for a genre
examples = few_shot_manager.get_examples("lovecraftian", n_examples=2)

# Display examples
for i, example in enumerate(examples, 1):
    print(f"Example {i}:")
    print("Original:")
    print(example['original'])
    print("\nStyled Version:")
    print(example['styled'])
    print("-" * 80)
```

## 3. Structured Output

### Story Generation with Structure

```python
from src.generator import NarrativeGenerator

generator = NarrativeGenerator()

# Example story summary
summary = """
A scientist discovers an ancient artifact in the Arctic that seems to defy the laws of physics.
"""

# Generate styled narrative with structured output
result = generator.generate_styled_narrative(
    summary=summary,
    target_genre="lovecraftian",
    style_examples=retrieved_examples,
    temperature=0.7
)

# Display structured result
print("Title:", result['title'])
print("\nAuthor's Note:", result['author_note'])
print("\nSections:")
for section in result['sections']:
    print(f"\n{section['title']}:")
    print(section['content'])

print("\nStyle Elements Used:")
for element in result['style_elements']:
    print(f"\n{element['element_type']}:")
    print(f"Description: {element['description']}")
    print("Examples:", ", ".join(element['examples']))
```

## 4. Function Calling

### Style Analysis Function

```python
from src.functions.style_analyzer import StyleAnalyzer

analyzer = StyleAnalyzer()

def analyze_style(text: str, genre: str = None) -> dict:
    """Analyze the style of a text."""
    return analyzer.analyze_text(text, genre)

# Example usage with function calling
styled_story = generator.transform_with_style_function(
    summary=summary,
    target_genre="lovecraftian",
    style_intensity=0.8,
    preserve_plot_elements=True
)

# Analyze the generated story
analysis = analyze_style(
    "\n".join([section['content'] for section in styled_story['sections']]),
    genre="lovecraftian"
)

print("Style Analysis Results:")
print(json.dumps(analysis, indent=2))
```

## 5. Command Line Demo

The project includes a command-line demo script (`run_demo.py`) that demonstrates the narrative style transfer capabilities:

```python
from src.main import NarrativeStyleTransferer
from src.data_processing import StyleCorpusProcessor
from src.vector_store import StyleVectorStore

def run_demo(show_technical_details: bool = False, initialize_db: bool = False):
    """Run the narrative style transfer demo.
    
    Args:
        show_technical_details: Show technical analysis of the generated story
        initialize_db: Initialize and populate the vector store
    """
    # Initialize vector store with optional corpus processing
    vector_store = StyleVectorStore(
        persist_directory="vector_store",
        use_existing=not initialize_db  # Use existing DB if not initializing
    )
    
    if initialize_db:
        # Process and add new documents
        corpus_processor = StyleCorpusProcessor()
        corpus = corpus_processor.load_corpus("style_corpus")
        processed_corpus = corpus_processor.create_embeddings(corpus)
        
        for genre, chunks in processed_corpus.items():
            vector_store.add_documents(chunks)
    
    # Initialize the transferer with vector store
    transferer = NarrativeStyleTransferer(
        vector_store=vector_store
    )
    
    # Example story transformation
    input_summary = (
        "A lone astronaut explores a newly discovered cave system on Mars. "
        "Inside, she finds strange, bioluminescent fungi and ancient carvings "
        "that depict unfamiliar constellations."
    )
    
    styled_story = transferer.transform_story(
        input_summary,
        target_genre="lovecraftian"
    )
    
    # Display the result
    print(f"\nTitle: {styled_story['title']}")
    print(f"\nAuthor's Note: {styled_story['author_note']}\n")
    
    for section in styled_story['sections']:
        print(f"\n## {section['title']} ##")
        print(section['content'])
```

### Running the Demo

You can run the demo with various options:

```bash
# Just generate a story (uses existing vector store)
python run_demo.py

# Initialize a new vector store and generate a story
python run_demo.py --init-db

# Show technical details about the generated story
python run_demo.py --show-technical

# Both initialize DB and show technical details
python run_demo.py --init-db --show-technical
```

Note: The vector store uses LlamaIndex with ChromaDB backend for persistent storage. This means your embeddings and documents are automatically saved and can be reused across sessions without needing to reinitialize the database each time.
```

## Results and Evaluation

Here we'll demonstrate the system's capabilities with a few example transformations:

1. **Original Summary**: A detective investigates a series of disappearances in a big city.
   - Transformed to Lovecraftian Horror
   - Transformed to Cyberpunk
   - Style analysis comparison

2. **Original Summary**: A young wizard discovers a mysterious spell book.
   - Transformed to Noir
   - Transformed to Cyberpunk
   - Style analysis comparison

[Include example outputs and analysis here]

## Conclusion

This project successfully demonstrates five key GenAI capabilities:

1. **RAG**: Effective use of retrieved examples to guide style transformation
2. **Vector Search**: Efficient retrieval of relevant style examples
3. **Few-Shot Prompting**: Clear demonstration of style through examples
4. **Structured Output**: Consistent and organized story generation
5. **Function Calling**: Dynamic style analysis and parameter adjustment

### Future Improvements

1. Enhanced style analysis using more sophisticated NLP techniques
2. Support for more genres and sub-genres
3. Better preservation of plot elements during transformation
4. Integration with other creative writing tools

### References

1. Google GenAI Documentation
2. Langchain Documentation
3. ChromaDB Documentation
4. Relevant academic papers on style transfer

---

To use this project:
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set your Google API key: `export GOOGLE_API_KEY='your-api-key'`
4. Run the demo: `python run_demo.py`

For development:
- Use `--init-db` flag to initialize the vector database
- Use `--show-technical` flag to see detailed style analysis

The complete code and documentation are available in this repository.
