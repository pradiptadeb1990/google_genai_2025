"""
Style analyzer function implementation for the Narrative Style Transferer.
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional


class StyleAnalyzer:
    """
    Analyzes and extracts style elements from text.
    Used for function calling within the Narrative Style Transferer.
    """

    def __init__(self):
        """Initialize the style analyzer."""
        # Genre-specific style markers
        self.genre_markers = {
            "fairy_tale": {
                "vocabulary": [
                    "once upon a time",
                    "kingdom",
                    "princess",
                    "prince",
                    "fairy",
                    "magic",
                    "enchanted",
                    "castle",
                    "witch",
                    "spell",
                    "happily ever after",
                    "wand",
                    "curse",
                    "magical",
                    "forest",
                    "queen",
                    "king",
                    "godmother",
                    "wishes",
                    "true love"
                ],
                "themes": [
                    "good vs evil",
                    "magical transformation",
                    "true love",
                    "moral lessons",
                    "enchantment",
                    "royal characters",
                    "magical helpers",
                    "happy endings",
                    "wishes and dreams",
                    "overcoming adversity"
                ]
            },
            "whimsical": {
                "vocabulary": [
                    "quirky",
                    "playful",
                    "silly",
                    "colorful",
                    "sparkle",
                    "twinkle",
                    "dance",
                    "giggle",
                    "flutter",
                    "bubble",
                    "rainbow",
                    "glitter",
                    "wonder",
                    "delight",
                    "mischief",
                    "charm",
                    "bounce",
                    "swirl",
                    "magical",
                    "dream"
                ],
                "themes": [
                    "playfulness",
                    "imagination",
                    "lighthearted fun",
                    "magical realism",
                    "childlike wonder",
                    "unexpected joy",
                    "gentle humor",
                    "everyday magic",
                    "whimsy",
                    "dreamlike elements"
                ]
            },
            "bedtime": {
                "vocabulary": [
                    "sleepy",
                    "dream",
                    "stars",
                    "moon",
                    "night",
                    "cozy",
                    "soft",
                    "quiet",
                    "gentle",
                    "lullaby",
                    "yawn",
                    "snuggle",
                    "peaceful",
                    "twilight",
                    "bedtime",
                    "cuddle",
                    "blanket",
                    "pillow",
                    "nightlight",
                    "goodnight"
                ],
                "themes": [
                    "bedtime routine",
                    "sweet dreams",
                    "comfort",
                    "peaceful sleep",
                    "nighttime adventures",
                    "gentle endings",
                    "family bonds",
                    "soothing atmosphere",
                    "safety and security",
                    "calming elements"
                ]
            },
            "lovecraftian": {
                "vocabulary": [
                    "eldritch",
                    "ancient",
                    "cyclopean",
                    "non-euclidean",
                    "madness",
                    "incomprehensible",
                    "cosmic",
                    "horror",
                    "dread",
                    "forbidden",
                    "blasphemous",
                    "nameless",
                    "crawling",
                    "chaos",
                    "tentacles",
                    "gibbering",
                    "insanity",
                    "abyss",
                    "terror",
                    "unspeakable",
                ],
                "themes": [
                    "cosmic horror",
                    "ancient beings",
                    "forbidden knowledge",
                    "insignificance of humanity",
                    "madness",
                    "isolation",
                    "unknown entities",
                    "dreams",
                    "ancient civilizations",
                ],
            },
            "cyberpunk": {
                "vocabulary": [
                    "neural",
                    "chrome",
                    "matrix",
                    "hack",
                    "augmented",
                    "cyber",
                    "implant",
                    "virtual",
                    "corporation",
                    "street",
                    "jacked",
                    "neon",
                    "digital",
                    "tech",
                    "wetware",
                    "synthetic",
                    "interface",
                    "megacorp",
                    "grid",
                    "wire",
                ],
                "themes": [
                    "technology",
                    "corporations",
                    "dystopia",
                    "artificial intelligence",
                    "virtual reality",
                    "transhumanism",
                    "cybernetic enhancement",
                    "hacking",
                    "social inequality",
                    "urban decay",
                ],
            },
            "noir": {
                "vocabulary": [
                    "dame",
                    "gumshoe",
                    "smoke",
                    "shadows",
                    "rain",
                    "whiskey",
                    "fedora",
                    "detective",
                    "case",
                    "murder",
                    "gun",
                    "dark",
                    "night",
                    "cigarette",
                    "alley",
                    "corrupt",
                    "mysterious",
                    "dangerous",
                    "city",
                    "double-cross",
                ],
                "themes": [
                    "crime",
                    "moral ambiguity",
                    "urban setting",
                    "detective work",
                    "femme fatale",
                    "corruption",
                    "betrayal",
                    "mystery",
                    "hard-boiled",
                    "pessimism",
                ],
            },
            "fantasy": {
                "vocabulary": [
                    "sword",
                    "magic",
                    "quest",
                    "dragon",
                    "spell",
                    "kingdom",
                    "prophecy",
                    "wizard",
                    "enchanted",
                    "mystical",
                    "ancient",
                    "hero",
                    "journey",
                    "creature",
                    "realm",
                    "power",
                    "scroll",
                    "destiny",
                    "legend",
                    "tome",
                ],
                "themes": [
                    "magic",
                    "mythical creatures",
                    "quests",
                    "alternate worlds",
                    "good vs. evil",
                    "prophecy",
                    "coming of age",
                    "heroism",
                    "adventure",
                    "royalty",
                ],
            },
        }

    def analyze_text(self, text: str, genre: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze text for style elements.

        Args:
            text: Text to analyze
            genre: Optional genre to check for specific markers

        Returns:
            Dictionary of style analysis results
        """
        results = {
            "vocabulary": self._analyze_vocabulary(text),
            "sentence_structure": self._analyze_sentence_structure(text),
            "themes": self._identify_themes(text),
        }

        # Check for genre-specific markers if genre is provided and it's valid
        if genre and genre.lower() in self.genre_markers:
            marker_results = self._check_genre_markers(text, genre.lower())
            results["genre_markers"] = marker_results
            results["genre_match_score"] = marker_results["match_score"]

        return results

    def _analyze_vocabulary(self, text: str) -> Dict[str, Any]:
        """
        Analyze the vocabulary characteristics of the text.


        Args:
            text: Text to analyze

        Returns:
            Dictionary of vocabulary analysis
        """
        # Clean and tokenize the text
        clean_text = re.sub(r"[^\w\s]", "", text.lower())
        words = clean_text.split()

        # Get word statistics [word count, unique words, word frequency, common words, average word length]
        word_count = len(words)
        unique_words = len(set(words))
        word_freq = Counter(words)
        common_words = word_freq.most_common(10)
        avg_word_length = (
            sum(len(word) for word in words) / word_count if word_count > 0 else 0
        )

        # Calculate lexical diversity (type-token ratio)
        lexical_diversity = unique_words / word_count if word_count > 0 else 0

        # Get uncommon words [words with length > 6 and frequency <= 3]
        uncommon_words = [
            word for word, count in word_freq.items() if len(word) > 6 and count <= 3
        ]

        # Return the vocabulary analysis results with common words and uncommon words (limited to first 10)
        return {
            "word_count": word_count,
            "unique_words": unique_words,
            "lexical_diversity": lexical_diversity,
            "avg_word_length": avg_word_length,
            "common_words": common_words,
            "uncommon_words": uncommon_words[:10],
        }

    def _analyze_sentence_structure(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentence structure characteristics of the text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary of sentence structure analysis
        """
        # Split into sentences
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Get sentence statistics
        sentence_count = len(sentences)
        if sentence_count == 0:
            return {
                "sentence_count": 0,
                "avg_sentence_length": 0,
                "sentence_length_variation": 0,
                "sentence_examples": [],
            }

        sentence_lengths = [len(s.split()) for s in sentences]
        avg_sentence_length = sum(sentence_lengths) / sentence_count

        # Measure variation in sentence length
        sentence_length_variation = (
            sum(abs(l - avg_sentence_length) for l in sentence_lengths) / sentence_count
        )

        # Example sentences (short, medium, long)
        sorted_sentences = sorted(zip(sentence_lengths, sentences))
        sentence_examples = []
        if sentence_count >= 3:
            # Get short, medium, and long examples
            sentence_examples = [
                {"type": "short", "text": sorted_sentences[0][1]},
                {
                    "type": "medium",
                    "text": sorted_sentences[len(sorted_sentences) // 2][1],
                },
                {"type": "long", "text": sorted_sentences[-1][1]},
            ]
        else:
            # Just use what we have
            for i, (length, sent) in enumerate(sorted_sentences):
                sent_type = "short" if i == 0 else "medium" if i == 1 else "long"
                sentence_examples.append({"type": sent_type, "text": sent})

        return {
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "sentence_length_variation": sentence_length_variation,
            "sentence_examples": sentence_examples,
        }

    def _identify_themes(self, text: str) -> List[str]:
        """
        Identify potential themes in the text.
        This is a simple keyword-based approach.

        Args:
            text: Text to analyze

        Returns:
            List of identified themes
        """
        # This is a simplified approach - in a real implementation,
        # this would use more sophisticated NLP techniques

        theme_keywords = {
            "adventure": ["journey", "quest", "discovery", "explore", "adventure"],
            "romance": ["love", "heart", "passion", "desire", "romance"],
            "mystery": ["mystery", "secret", "clue", "enigma", "puzzle"],
            "horror": ["fear", "terror", "dread", "horror", "nightmare"],
            "conflict": ["battle", "war", "fight", "conflict", "struggle"],
            "transformation": [
                "change",
                "transform",
                "evolve",
                "growth",
                "development",
            ],
            "good_vs_evil": ["good", "evil", "hero", "villain", "dark", "light"],
        }

        text_lower = text.lower()
        identified_themes = []

        for theme, keywords in theme_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches >= 2:  # Require at least 2 keywords to identify a theme
                identified_themes.append(theme)

        return identified_themes

    def _check_genre_markers(self, text: str, genre: str) -> Dict[str, Any]:
        """
        Check for genre-specific markers in the text.

        Args:
            text: Text to analyze
            genre: Genre to check markers for

        Returns:
            Dictionary of genre marker analysis
        """
        if genre not in self.genre_markers:
            return {"match_score": 0, "vocabulary_matches": [], "theme_matches": []}

        text_lower = text.lower()
        markers = self.genre_markers[genre]

        # Check vocabulary matches
        vocabulary_matches = []
        for word in markers["vocabulary"]:
            if word in text_lower:
                vocabulary_matches.append(word)

        # Check theme matches
        theme_matches = []
        for theme in markers["themes"]:
            theme_keywords = theme.split()
            if any(keyword in text_lower for keyword in theme_keywords):
                theme_matches.append(theme)

        # Calculate match score (0.0 to 1.0)
        vocab_score = (
            len(vocabulary_matches) / len(markers["vocabulary"])
            if markers["vocabulary"]
            else 0
        )
        theme_score = (
            len(theme_matches) / len(markers["themes"]) if markers["themes"] else 0
        )
        match_score = (vocab_score * 0.6) + (
            theme_score * 0.4
        )  # Weight vocabulary more

        return {
            "match_score": match_score,
            "vocabulary_matches": vocabulary_matches,
            "theme_matches": theme_matches,
        }
