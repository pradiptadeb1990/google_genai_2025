"""
Vector database operations for the Narrative Style Transferer.
Handles storing and retrieving embeddings for style examples.
"""

import json
import os
from typing import Any, Dict, List, Optional

# We'll use Chroma as our vector database
import chromadb
import numpy as np
from chromadb.utils import embedding_functions
from tqdm import tqdm


class StyleVectorDB:
    """
    Vector database for style examples using ChromaDB.
    """

    def __init__(self, persist_directory: str):
        """
        Initialize the vector database.

        Args:
            persist_directory: Directory to store the vector database
        """
        self.persist_directory = persist_directory
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Use sentence-transformers for embedding consistency
        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )

    def create_collection(
        self, collection_name: str = "style_examples"
    ) -> chromadb.Collection:
        """
        Create a collection in the vector database.

        Args:
            collection_name: Name of the collection to create

        Returns:
            ChromaDB collection
        """
        # Check if collection already exists
        existing_collections = self.client.list_collections()
        for collection in existing_collections:
            if collection.name == collection_name:
                return self.client.get_collection(
                    name=collection_name, embedding_function=self.embedding_function
                )

        # Create new collection
        return self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "Style examples for narrative style transfer"},
        )

    def add_documents(
        self, documents: List[Dict[str, Any]], collection_name: str = "style_examples"
    ):
        """
        Add documents to the vector database.

        Args:
            documents: List of document dictionaries with 'text', 'embedding', and 'metadata'
            collection_name: Name of the collection to add to
        """
        collection = self.create_collection(collection_name)

        # Prepare data for batch addition
        ids = []
        texts = []
        embeddings = []
        metadatas = []

        for i, doc in enumerate(documents):
            doc_id = f"{doc['metadata']['genre']}_{doc['metadata']['source_file']}_{doc['metadata']['chunk_id']}"
            ids.append(doc_id)
            texts.append(doc["text"])
            # Use pre-computed embeddings if available, otherwise let ChromaDB compute them
            if "embedding" in doc:
                embeddings.append(doc["embedding"])
            metadatas.append(doc["metadata"])

        # Add documents in batches to avoid memory issues
        batch_size = 100
        for i in tqdm(
            range(0, len(ids), batch_size), desc=f"Adding to {collection_name}"
        ):
            batch_end = min(i + batch_size, len(ids))

            batch_ids = ids[i:batch_end]
            batch_texts = texts[i:batch_end]
            batch_metadatas = metadatas[i:batch_end]

            if embeddings:
                batch_embeddings = embeddings[i:batch_end]
                collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                )
            else:
                collection.add(
                    ids=batch_ids, documents=batch_texts, metadatas=batch_metadatas
                )

    def load_from_jsonl(self, jsonl_path: str, collection_name: str = "style_examples"):
        """
        Load documents from a JSONL file into the vector database.

        Args:
            jsonl_path: Path to the JSONL file
            collection_name: Name of the collection to add to
        """
        documents = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                documents.append(doc)

        self.add_documents(documents, collection_name)

    def query(
        self,
        query_text: str,
        collection_name: str = "style_examples",
        n_results: int = 5,
        genre_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query the vector database for similar style examples.

        Args:
            query_text: Query text
            collection_name: Name of the collection to query
            n_results: Number of results to return
            genre_filter: Optional filter for specific genre

        Returns:
            List of similar style examples
        """
        collection = self.create_collection(collection_name)

        # Prepare filter if needed
        where_filter = {}
        if genre_filter:
            where_filter = {"genre": genre_filter}

        # Query the database
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter if where_filter else None,
        )

        # Format results
        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append(
                {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": (
                        results["distances"][0][i] if "distances" in results else None
                    ),
                }
            )

        return formatted_results

    def query_by_genre(
        self,
        query_text: str,
        target_genre: str,
        collection_name: str = "style_examples",
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Query for examples of a specific genre that match the query.

        Args:
            query_text: Query text
            target_genre: Target genre to filter by
            collection_name: Name of the collection to query
            n_results: Number of results to return

        Returns:
            List of similar style examples in the target genre
        """
        return self.query(
            query_text, collection_name, n_results, genre_filter=target_genre
        )

    def get_genre_examples(
        self, genre: str, collection_name: str = "style_examples", n_examples: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get random examples from a specific genre.

        Args:
            genre: Genre to get examples from
            collection_name: Name of the collection to query
            n_examples: Number of examples to return

        Returns:
            List of style examples from the specified genre
        """
        collection = self.create_collection(collection_name)

        # Query by genre without a specific query text
        results = collection.get(where={"genre": genre}, limit=n_examples)

        # Format results
        formatted_results = []
        for i in range(len(results["documents"])):
            formatted_results.append(
                {"text": results["documents"][i], "metadata": results["metadatas"][i]}
            )

        return formatted_results
