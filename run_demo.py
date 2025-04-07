import json
import os
import sys

# Add the src directory to the Python path
# This allows importing modules from src like src.main
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from data_processing import StyleCorpusProcessor
    from main import NarrativeStyleTransferer
    from vector_db import StyleVectorDB
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you are running this script from the project root directory")
    print("and that all dependencies from requirements.txt are installed.")
    sys.exit(1)


def run_demo(show_technical_details: bool = False, initialize_db: bool = False):
    """Runs a demonstration of the Narrative Style Transferer.

    Args:
        show_technical_details (bool): Whether to show technical analysis of the generated story
        initialize_db (bool): Whether to initialize and populate the vector database
    """
    # --- Configuration ---
    # Ensure the Google API Key is set as an environment variable
    # You can set it in your terminal before running the script:
    # export GOOGLE_API_KEY='YOUR_API_KEY'
    # Or, uncomment and set it directly here (NOT RECOMMENDED for security):
    # os.environ['GOOGLE_API_KEY'] = 'YOUR_API_KEY'

    if "GOOGLE_API_KEY" not in os.environ:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set it before running the script.")
        print("Example: export GOOGLE_API_KEY='YOUR_API_KEY'")
        sys.exit(1)

    # Paths (relative to project root)
    data_dir = os.path.join(project_root, "data")

    # --- Initialization ---
    print("Initializing Narrative Style Transferer...")

    # Initialize vector database with optional corpus processing
    print("\n--- Setting up Vector Database ---")
    vector_db_dir = os.path.join(data_dir, "vector_db_store")
    vector_db = StyleVectorDB(persist_directory=vector_db_dir)

    if initialize_db:
        print("Processing style corpus...")
        corpus_processor = StyleCorpusProcessor()
        corpus = corpus_processor.load_corpus(os.path.join(data_dir, "style_corpus"))
        processed_corpus = corpus_processor.create_embeddings(corpus)

        # Add to vector database
        print("Adding documents to vector database...")
        for genre, chunks in processed_corpus.items():
            vector_db.add_documents(chunks)
            print(f"Added {len(chunks)} chunks for genre: {genre}")
        print("Vector database initialization complete.")
    else:
        print("Using existing vector database...")

    try:
        # Initialize transferer
        transferer = NarrativeStyleTransferer(
            data_dir=data_dir, api_key=os.environ.get("GOOGLE_API_KEY")
        )
        print("Initialization complete.")
    except Exception as e:
        print(f"Error during initialization: {e}")
        sys.exit(1)

    # --- Define Input ---
    input_summary = (
        "A lone astronaut explores a newly discovered cave system on Mars. "
        "Inside, she finds strange, bioluminescent fungi and ancient carvings "
        "that depict unfamiliar constellations. Her communication systems start failing "
        "as she ventures deeper into the dark, winding tunnels."
    )
    target_genre = "fairy_tale"  # Try changing this to: cyberpunk, noir, fairy_tale, whimsical, bedtime

    print(f"\n--- Input Summary ---")
    print(input_summary)
    print(f"\n--- Target Genre ---")
    print(target_genre)

    # --- Perform Style Transfer ---
    print(f"\n--- Transforming story to {target_genre} style ---")
    try:
        styled_story_dict = transferer.transform_story(input_summary, target_genre)

        # Extract and format the story content
        print("\n=== Generated Story ===\n")
        print(f"Title: {styled_story_dict.get('title', '')}")
        print(f"\nAuthor's Note: {styled_story_dict.get('author_note', '')}\n")

        # Print each section
        for section in styled_story_dict.get("sections", []):
            print(f"\n## {section.get('title', '')} ##")
            print(f"{section.get('content', '')}")

        # Print style analysis and metadata if flag is set
        if show_technical_details:
            print("\n=== Technical Details ===\n")

            # Style elements
            print("Style Elements:")
            for element in styled_story_dict.get("style_elements", []):
                print(f"\n- {element.get('element_type', '')}:")
                print(f"  {element.get('description', '')}")
                print("  Examples:")
                for example in element.get("examples", []):
                    print(f"    * {example}")

            # If style_analysis exists, print relevant metrics
            if "style_analysis" in styled_story_dict:
                analysis = styled_story_dict["style_analysis"]
                print("\nStyle Analysis:")
                if "vocabulary" in analysis:
                    vocab = analysis["vocabulary"]
                    print(f"- Word Count: {vocab.get('word_count', 0)}")
                    print(f"- Unique Words: {vocab.get('unique_words', 0)}")
                    print(
                        f"- Lexical Diversity: {vocab.get('lexical_diversity', 0):.2f}"
                    )

                if "genre_markers" in analysis:
                    markers = analysis["genre_markers"]
                    print(f"\nGenre Match Score: {markers.get('match_score', 0):.2f}")
                    print(
                        "Genre Vocabulary Matches:",
                        ", ".join(markers.get("vocabulary_matches", [])),
                    )

    except Exception as e:
        print(f"\n--- Error during transformation ---")
        print(f"An error occurred: {e}")
        # Add more specific error handling if needed

    print("\n--- Demo Finished ---")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Narrative Style Transferer demo"
    )
    parser.add_argument(
        "--show-technical",
        action="store_true",
        help="Show technical details of the generated story",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize and populate the vector database",
    )

    args = parser.parse_args()

    run_demo(show_technical_details=args.show_technical, initialize_db=args.init_db)
