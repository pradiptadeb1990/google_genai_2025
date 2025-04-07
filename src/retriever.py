"""
Retrieval component for the Narrative Style Transferer.
Handles retrieving relevant style examples for augmenting generation.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from src.data_processing import FewShotExampleManager
from src.vector_db import StyleVectorDB


class StyleRetriever:
    """
    Retrieves relevant style examples for the narrative style transfer process.
    Combines vector search with few-shot examples.
    """

    def __init__(
        self,
        vector_db: StyleVectorDB,
        few_shot_manager: FewShotExampleManager,
        collection_name: str = "style_examples",
    ):
        """
        Initialize the style retriever.

        Args:
            vector_db: Vector database instance
            few_shot_manager: Few-shot example manager instance
            collection_name: Name of the collection to query
        """
        self.vector_db = vector_db
        self.few_shot_manager = few_shot_manager
        self.collection_name = collection_name

    def retrieve_style_examples(
        self,
        query_text: str,
        target_genre: str,
        n_vector_results: int = 3,
        n_few_shot: int = 2,
    ) -> Dict[str, Any]:
        """
        Retrieve style examples for the given query and target genre.

        Args:
            query_text: User query or story summary
            target_genre: Target genre for style transformation
            n_vector_results: Number of vector search results to retrieve
            n_few_shot: Number of few-shot examples to include

        Returns:
            Dictionary containing both vector search results and few-shot examples
        """

        # Get semantically similar examples from the vector database
        vector_results = self.vector_db.query_by_genre(
            query_text=query_text,
            target_genre=target_genre,
            collection_name=self.collection_name,
            n_results=n_vector_results,
        )

        # Get curated few-shot examples
        few_shot_examples = self.few_shot_manager.get_examples(
            genre=target_genre, n_examples=n_few_shot
        )

        return {
            "vector_results": vector_results,
            "few_shot_examples": few_shot_examples,
            "target_genre": target_genre,
            "query": query_text,
        }

    def analyze_style_elements(self, genre: str) -> Dict[str, Any]:
        """
        Analyze and extract key style elements for a specific genre.

        Args:
            genre: Genre to analyze

        Returns:
            Dictionary of style elements characteristic of the genre
        """
        # Get random examples from the genre
        examples = self.vector_db.get_genre_examples(
            genre=genre, collection_name=self.collection_name, n_examples=5
        )

        # This is a placeholder for a more sophisticated style analysis
        # In a real implementation, this would use NLP techniques to extract
        # vocabulary, sentence structure, thematic elements, etc.
        style_elements = {
            "genre": genre,
            "example_count": len(examples),
            "common_themes": self._extract_themes(examples, genre),
            "vocabulary": self._extract_vocabulary(examples, genre),
            "sentence_structure": self._extract_sentence_structure(examples, genre),
        }

        return style_elements

    def format_for_prompt(self, retrieved_data: Dict[str, Any]) -> str:
        """
        Format retrieved data for inclusion in a prompt.

        Args:
            retrieved_data: Data returned by retrieve_style_examples

        Returns:
            Formatted string for inclusion in prompt
        """
        prompt_parts = [
            f"# Style Examples for {retrieved_data['target_genre']} Genre\n"
        ]

        # Add few-shot examples
        if retrieved_data["few_shot_examples"]:
            prompt_parts.append("## Transformation Examples\n")
            for i, example in enumerate(retrieved_data["few_shot_examples"]):
                prompt_parts.append(f"### Example {i+1}:\n")
                prompt_parts.append("Original:")
                prompt_parts.append(f"{example['original']}\n")
                prompt_parts.append(
                    "Transformed to " + retrieved_data["target_genre"] + ":"
                )
                prompt_parts.append(f"{example['styled']}\n")

        # Add vector search results
        if retrieved_data["vector_results"]:
            prompt_parts.append(
                f"## Style Reference Texts for {retrieved_data['target_genre']}\n"
            )
            for i, result in enumerate(retrieved_data["vector_results"]):
                prompt_parts.append(f"### Reference {i+1}:\n")
                prompt_parts.append(f"{result['text']}\n")
                if result["metadata"].get("author"):
                    prompt_parts.append(f"Author: {result['metadata']['author']}\n")

        return "\n".join(prompt_parts)

    def _extract_themes(self, examples: List[Dict[str, Any]], genre: str) -> List[str]:
        """
        Extract common themes from examples of a genre.
        This is a placeholder for more sophisticated analysis.

        Args:
            examples: List of text examples
            genre: Genre being analyzed

        Returns:
            List of common themes
        """
        # This would be implemented with more sophisticated NLP in a real app
        genre_themes = {
            "lovecraftian": [
                "cosmic horror",
                "unknown entities",
                "madness",
                "ancient beings",
                "forbidden knowledge",
            ],
            "cyberpunk": [
                "technology",
                "corporations",
                "dystopia",
                "artificial intelligence",
                "virtual reality",
            ],
            "noir": [
                "crime",
                "moral ambiguity",
                "urban setting",
                "detective",
                "femme fatale",
            ],
            "romance": [
                "love",
                "relationships",
                "emotional connection",
                "personal growth",
                "happy ending",
            ],
            "fantasy": [
                "magic",
                "mythical creatures",
                "quests",
                "alternate worlds",
                "good vs. evil",
            ],
            "western": [
                "frontier",
                "justice",
                "lawlessness",
                "rugged individualism",
                "wilderness",
            ],
        }

        return genre_themes.get(genre.lower(), ["theme1", "theme2", "theme3"])

    def _extract_vocabulary(
        self, examples: List[Dict[str, Any]], genre: str
    ) -> List[str]:
        """
        Extract characteristic vocabulary from examples of a genre.
        This is a placeholder for more sophisticated analysis.

        Args:
            examples: List of text examples
            genre: Genre being analyzed

        Returns:
            List of characteristic vocabulary words
        """
        # This would be implemented with more sophisticated NLP in a real app
        genre_vocabulary = {
            "lovecraftian": [
                "eldritch",
                "cosmic",
                "ancient",
                "madness",
                "incomprehensible",
                "cyclopean",
                "non-euclidean",
            ],
            "cyberpunk": [
                "neural",
                "chrome",
                "deck",
                "matrix",
                "corporation",
                "hack",
                "implant",
            ],
            "noir": [
                "dame",
                "gumshoe",
                "smoke",
                "shadows",
                "rain",
                "whiskey",
                "fedora",
            ],
            "romance": [
                "heart",
                "love",
                "passion",
                "desire",
                "embrace",
                "yearning",
                "intimacy",
            ],
            "fantasy": [
                "sword",
                "magic",
                "quest",
                "dragon",
                "spell",
                "kingdom",
                "prophecy",
            ],
            "western": [
                "sheriff",
                "outlaw",
                "saloon",
                "desert",
                "horse",
                "gun",
                "frontier",
            ],
        }

        return genre_vocabulary.get(genre.lower(), ["word1", "word2", "word3"])

    def _extract_sentence_structure(
        self, examples: List[Dict[str, Any]], genre: str
    ) -> List[str]:
        """
        Extract characteristic sentence structures from examples of a genre.
        This is a placeholder for more sophisticated analysis.

        Args:
            examples: List of text examples
            genre: Genre being analyzed

        Returns:
            List of characteristic sentence structures
        """
        # This would be implemented with more sophisticated NLP in a real app
        genre_structures = {
            "lovecraftian": [
                "long, complex sentences with many clauses",
                "archaic language",
                "first-person narration focusing on emotions of terror and awe",
            ],
            "cyberpunk": [
                "short, punchy sentences",
                "technical jargon",
                "present tense narration",
                "street slang",
            ],
            "noir": [
                "terse descriptions",
                "metaphors and similes",
                "first-person hardboiled narration",
            ],
            "romance": [
                "emotional descriptions",
                "sensory details",
                "dialogue rich in subtext",
            ],
            "fantasy": [
                "descriptive passages",
                "formal dialogue",
                "world-building exposition",
            ],
            "western": [
                "sparse dialogue",
                "environmental descriptions",
                "action-focused narration",
            ],
        }

        return genre_structures.get(genre.lower(), ["structure1", "structure2"])
