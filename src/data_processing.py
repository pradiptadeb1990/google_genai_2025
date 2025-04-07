"""
Data processing module for the Narrative Style Transferer.
Handles text processing and embedding generation.
"""

import json
import os
from typing import Any, Dict, List

import torch
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm


class StyleCorpusProcessor:
    """
    Process text examples from different literary styles and prepare them for embedding.
    Uses LlamaIndex for efficient document processing and retrieval.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = None,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
    ):
        """
        Initialize the processor with a sentence transformer model.

        Args:
            model_name: Name of the sentence transformer model to use for embeddings
            device: Device to use for computation ('cuda' for NVIDIA GPU, 'mps' for Apple Silicon, 'cpu' for CPU, None for auto)
        """
        if device is None:
            # Auto-detect the best available device
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # Initialize the embedding model
        embed_model = HuggingFaceEmbedding(model_name=model_name, device=device)

        # Configure LlamaIndex settings
        Settings.embed_model = embed_model
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap

        # Initialize the node parser for text chunking
        self.node_parser = SentenceSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        self.device = device
        print(f"Using device: {self.device}")

    def load_corpus(self, corpus_dir: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load style corpus from directory structure.

        Args:
            corpus_dir: Path to the directory containing style examples

        Returns:
            Dictionary mapping genre names to lists of text examples
        """
        corpus = {}

        for genre in os.listdir(corpus_dir):
            genre_path = os.path.join(corpus_dir, genre)
            if not os.path.isdir(genre_path):
                continue

            corpus[genre] = []

            for filename in os.listdir(genre_path):
                if not filename.endswith(".txt"):
                    continue

                file_path = os.path.join(genre_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                # Extract metadata from filename if available
                # Assuming format: author_title.txt
                metadata = {}
                if "_" in filename:
                    author = filename.split("_")[0]
                    title = filename.split("_")[1].replace(".txt", "")
                    metadata = {"author": author, "title": title}

                corpus[genre].append(
                    {"text": text, "filename": filename, "metadata": metadata}
                )

        return corpus

    def chunk_text(
        self, text: str, chunk_size: int = 300, overlap: int = 50
    ) -> List[str]:
        """
        Split text into overlapping chunks for better retrieval.

        Args:
            text: Text to chunk
            chunk_size: Maximum number of characters per chunk
            overlap: Number of characters to overlap between chunks

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            # Find a good break point (sentence end)
            end = min(start + chunk_size, len(text))

            # Try to find a sentence break
            if end < len(text):
                sentence_breaks = [".", "!", "?", "\n\n"]
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in sentence_breaks:
                        end = i + 1
                        break

            chunks.append(text[start:end])
            start = end - overlap

        return chunks

    def create_embeddings(
        self, corpus: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create embeddings for all chunks in the corpus using LlamaIndex.

        Args:
            corpus: Dictionary mapping genre names to lists of text examples

        Returns:
            Corpus with added embeddings for each chunk
        """
        processed_corpus = {}

        for genre, documents in corpus.items():
            processed_corpus[genre] = []
            print(f"\nProcessing documents for genre: {genre}")

            # Convert documents to LlamaIndex format
            llama_docs = []
            for doc in tqdm(documents, desc="Converting documents"):
                metadata = {
                    **doc["metadata"],
                    "genre": genre,
                    "source_file": doc["filename"],
                }

                llama_doc = Document(text=doc["text"], metadata=metadata)
                llama_docs.append(llama_doc)

            # Create index and process documents
            index = VectorStoreIndex.from_documents(llama_docs, show_progress=True)

            # Get the storage context which contains the embeddings
            storage_context = index.storage_context

            # Extract processed nodes with embeddings
            for node_id in index.index_struct.nodes_dict:
                # Get the node from storage
                node = storage_context.docstore.get_node(node_id)

                # Get embedding from vector store
                embedding = storage_context.vector_store.get(node_id)

                processed_corpus[genre].append(
                    {
                        "text": node.text,
                        "embedding": embedding,
                        "metadata": {
                            **node.metadata,
                            "chunk_id": node_id,
                        },
                    }
                )

            print(f"Processed {len(processed_corpus[genre])} chunks for {genre}")

        return processed_corpus

    def save_processed_corpus(
        self, processed_corpus: Dict[str, List[Dict[str, Any]]], output_dir: str
    ):
        """
        Save processed corpus to disk.

        Args:
            processed_corpus: Processed corpus with embeddings
            output_dir: Directory to save processed corpus
        """
        os.makedirs(output_dir, exist_ok=True)

        for genre, chunks in processed_corpus.items():
            genre_dir = os.path.join(output_dir, genre)
            os.makedirs(genre_dir, exist_ok=True)

            # Save as jsonl for easier loading
            with open(
                os.path.join(genre_dir, f"{genre}_chunks.jsonl"), "w", encoding="utf-8"
            ) as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk) + "\n")

            print(f"Saved {len(chunks)} chunks for genre: {genre}")


class FewShotExampleManager:
    """
    Manage few-shot examples for style transformation.
    """

    def __init__(self, examples_dir: str):
        """
        Initialize the manager with path to examples directory.

        Args:
            examples_dir: Path to the directory containing few-shot examples
        """
        self.examples_dir = examples_dir
        self.examples = self._load_examples()

    def _load_examples(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Load few-shot examples from disk.

        Returns:
            Dictionary mapping genre names to lists of examples
        """
        examples = {}

        if not os.path.exists(self.examples_dir):
            return examples

        for filename in os.listdir(self.examples_dir):
            if not filename.endswith(".json"):
                continue

            genre = filename.replace(".json", "")
            file_path = os.path.join(self.examples_dir, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                genre_data = json.load(f)

            # Ensure consistent structure
            if isinstance(genre_data, list):
                # If it's already a list of examples, wrap it
                examples[genre] = {"examples": genre_data}
            elif isinstance(genre_data, dict) and "examples" in genre_data:
                # If it has the expected structure, use as is
                examples[genre] = genre_data
            else:
                # If it's in some other format, wrap it in our structure
                examples[genre] = {"examples": []}

        return examples

    def get_examples(self, genre: str, n_examples: int = 3) -> List[Dict[str, str]]:
        """
        Get examples for a specific genre.

        Args:
            genre: Genre to get examples for
            n_examples: Number of examples to return

        Returns:
            List of example dictionaries with 'original' and 'styled' keys
        """
        if genre not in self.examples:
            return []

        # Handle nested structure where examples are under 'examples' key
        examples_list = self.examples[genre].get("examples", [])
        return examples_list[:n_examples]

    def create_example(self, genre: str, original: str, styled: str):
        """
        Create a new few-shot example.

        Args:
            genre: Genre of the example
            original: Original text
            styled: Styled version of the text
        """
        if genre not in self.examples:
            self.examples[genre] = {"examples": []}

        self.examples[genre].append({"original": original, "styled": styled})

        # Save to disk
        os.makedirs(self.examples_dir, exist_ok=True)
        file_path = os.path.join(self.examples_dir, f"{genre}.json")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.examples[genre], f, indent=2)
