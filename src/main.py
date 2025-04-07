"""
Main application module for the Narrative Style Transferer.
Ties together all components and provides a simple API.
"""

import json
import os
from typing import Any, Dict, List, Optional

from src.data_processing import FewShotExampleManager, StyleCorpusProcessor
from src.functions.style_analyzer import StyleAnalyzer
from src.generator import NarrativeGenerator
from src.retriever import StyleRetriever
from src.vector_db import StyleVectorDB


class NarrativeStyleTransferer:
    """
    Main application class for the Narrative Style Transferer.
    """

    def __init__(self, data_dir: str, api_key: Optional[str] = None):
        """
        Initialize the Narrative Style Transferer.

        Args:
            data_dir: Directory containing data files
            api_key: Google AI API key (if None, will look for env variable)
        """
        self.data_dir = data_dir

        # Paths for data components
        corpus_dir = os.path.join(data_dir, "style_corpus")
        few_shot_dir = os.path.join(data_dir, "few_shot_examples")
        vector_store_dir = os.path.join(data_dir, "vector_store")

        # Create directories if they don't exist
        for directory in [corpus_dir, few_shot_dir, vector_store_dir]:
            os.makedirs(directory, exist_ok=True)

        # Initialize components
        self.corpus_processor = StyleCorpusProcessor()
        self.vector_db = StyleVectorDB(vector_store_dir)
        self.few_shot_manager = FewShotExampleManager(few_shot_dir)
        self.retriever = StyleRetriever(self.vector_db, self.few_shot_manager)
        self.generator = NarrativeGenerator(api_key)
        self.style_analyzer = StyleAnalyzer()

        # Available genres (will be populated when corpus is processed)
        self.available_genres = []
        self._load_available_genres()

    def _load_available_genres(self):
        """Load available genres from the corpus directory."""
        corpus_dir = os.path.join(self.data_dir, "style_corpus")
        if os.path.exists(corpus_dir):
            self.available_genres = [
                d
                for d in os.listdir(corpus_dir)
                if os.path.isdir(os.path.join(corpus_dir, d))
            ]

    def process_corpus(self):
        """Process the style corpus and store embeddings in the vector database."""
        corpus_dir = os.path.join(self.data_dir, "style_corpus")

        # Load and process corpus
        corpus = self.corpus_processor.load_corpus(corpus_dir)
        processed_corpus = self.corpus_processor.create_embeddings(corpus)

        # Save processed corpus and add to vector database
        for genre, chunks in processed_corpus.items():
            # Save to disk
            genre_dir = os.path.join(corpus_dir, genre, "processed")
            os.makedirs(genre_dir, exist_ok=True)

            output_file = os.path.join(genre_dir, f"{genre}_chunks.jsonl")
            with open(output_file, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk) + "\n")

            # Add to vector database
            self.vector_db.add_documents(chunks)

        # Update available genres
        self._load_available_genres()

        return len(processed_corpus)

    def transform_story(
        self,
        summary: str,
        target_genre: str,
        temperature: float = 0.7,
        use_function_calling: bool = True,
    ) -> Dict[str, Any]:
        """
        Transform a story summary into a specific genre style.

        Args:
            summary: Summary of the story to transform
            target_genre: Target genre for the transformation
            temperature: Creativity parameter (0.0-1.0)
            use_function_calling: Whether to use function calling approach

        Returns:
            Transformed story in structured format
        """

        # Retrieve style examples
        retrieved_data = self.retriever.retrieve_style_examples(
            query_text=summary, target_genre=target_genre
        )

        # Format examples for the prompt
        style_examples = self.retriever.format_for_prompt(retrieved_data)
        # Generate the transformed story
        if use_function_calling:
            # Use function calling approach
            transformed_story = self.generator.transform_with_style_function(
                summary=summary, target_genre=target_genre, style_intensity=temperature
            )
        else:
            # Use standard approach with structured output
            transformed_story = self.generator.generate_styled_narrative(
                summary=summary,
                target_genre=target_genre,
                style_examples=style_examples,
                temperature=temperature,
            )

        # Add style analysis
        if transformed_story.get("sections"):
            story_text = "\n\n".join(
                [section["content"] for section in transformed_story["sections"]]
            )
            transformed_story["style_analysis"] = self.style_analyzer.analyze_text(
                text=story_text, genre=target_genre
            )

        return transformed_story

    def get_available_genres(self) -> List[str]:
        """Get list of available genres in the corpus."""
        return self.available_genres

    def analyze_style(self, text: str, genre: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze the style of a text.

        Args:
            text: Text to analyze
            genre: Optional genre to check against

        Returns:
            Style analysis results
        """
        return self.style_analyzer.analyze_text(text, genre)
