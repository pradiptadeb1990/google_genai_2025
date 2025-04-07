# Narrative Style Transferer - Google GenAI Intensive Course Capstone

This project implements a system that transforms story summaries into specific narrative styles (e.g., turning a sci-fi story summary into Lovecraftian horror) using advanced GenAI techniques.

## GenAI Capabilities Demonstrated

1. **Retrieval Augmented Generation (RAG)** - Core architecture for style transfer
2. **Vector Search/Vector Database** - For efficient style example retrieval
3. **Few-Shot Prompting** - To guide style transformation with examples
4. **Structured Output** - For consistent story generation with defined sections
5. **Function Calling** - For dynamic style selection and narrative processing

## Project Structure

```
project/
├── requirements.txt       # Dependencies
├── README.md             # Project documentation
├── data/
│   ├── style_corpus/     # Style examples organized by genre
│   ├── few_shot_examples/# Curated examples for prompting
│   └── vector_store/     # Embedded style examples
├── src/
│   ├── data_processing.py# Text processing and embedding
│   ├── vector_db.py      # Vector database operations
│   ├── retriever.py      # RAG retrieval component
│   ├── few_shot.py       # Few-shot example management
│   ├── generator.py      # Structured text generation
│   ├── functions/        # Function calling implementations
│   │   ├── style_analyzer.py
│   │   ├── genre_classifier.py
│   │   └── parameter_tuner.py
│   ├── eval.py           # Evaluation utilities
│   └── main.py           # Main application
└── notebooks/
    └── narrative_style_transferer.ipynb # Main Kaggle submission notebook
```

## Setup

1. Install requirements:
```
pip install -r requirements.txt
```

2. Configure API access:
Add your API key to the appropriate configuration file or environment variable.

3. Run the Kaggle notebook:
The complete demo is available in `notebooks/narrative_style_transferer.ipynb`

## Kaggle Submission

This project was developed as a capstone for the Google GenAI Intensive Course 2025. The final submission is a Kaggle notebook demonstrating the capabilities of the Narrative Style Transferer.
