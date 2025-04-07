"""
Generator component for the Narrative Style Transferer.
Handles generating styled narratives using the Gemini API with structured output.
"""

import json
import os
from typing import Any, Dict, List, Optional, Union

import google.generativeai as genai
from google.ai.generativelanguage import (FunctionDeclaration, Schema, Tool,
                                          Type)
from pydantic import BaseModel, Field


# Define structured output models for function calling
class StorySection(BaseModel):
    """Model for a section of a story."""

    title: str = Field(..., description="Title of this section")
    content: str = Field(..., description="Content of this section")


class StyleElement(BaseModel):
    """Model for a style element used in the story."""

    element_type: str = Field(
        ..., description="Type of style element (vocabulary, structure, theme, etc.)"
    )
    description: str = Field(..., description="Description of the style element")
    examples: List[str] = Field(
        ..., description="Examples of this style element in the generated text"
    )


class StyledStory(BaseModel):
    """Model for the complete styled story output."""

    title: str = Field(..., description="Title of the styled story")
    genre: str = Field(..., description="Genre/style of the story")
    author_note: str = Field(..., description="Brief note about the style and approach")
    sections: List[StorySection] = Field(..., description="Sections of the story")
    style_elements: List[StyleElement] = Field(
        ..., description="Style elements used in the story"
    )


class NarrativeGenerator:
    """
    Generates styled narratives using the Gemini API.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the narrative generator.

        Args:
            api_key: Google AI API key. If None, will look for GOOGLE_API_KEY env variable.
        """
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")

        if api_key is None:
            raise ValueError(
                "No API key provided. Set GOOGLE_API_KEY environment variable or pass api_key."
            )

        # Configure the Gemini API
        genai.configure(api_key=api_key)

        # Use Gemini Pro model
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def generate_styled_narrative(
        self,
        summary: str,
        target_genre: str,
        style_examples: str,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Generate a styled narrative from a summary.

        Args:
            summary: Summary of the story
            target_genre: Target genre for styling
            style_examples: Style examples from the retriever
            temperature: Creativity parameter (0.0-1.0)

        Returns:
            Structured story output
        """
        # Create prompt for structured output
        prompt = self._create_prompt(summary, target_genre, style_examples)

        # Generate with structured output
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                top_p=0.95,
                top_k=40,
                candidate_count=1,
                max_output_tokens=4096,
            ),
            # Request structured format
            response_mime_type="application/json",
        )

        # Parse the response into our structured format
        try:
            structured_story = json.loads(response.text)
            # Validate against our schema
            return StyledStory(**structured_story).dict()
        except Exception as e:
            # Fallback to unstructured response if parsing fails
            return {
                "title": "Generated Story",
                "genre": target_genre,
                "author_note": "Note: This story was generated with style transfer.",
                "sections": [{"title": "Story", "content": response.text}],
                "style_elements": [],
                "error": str(e),
            }

    def _create_prompt(
        self, summary: str, target_genre: str, style_examples: str
    ) -> str:
        """
        Create a prompt for the Gemini API.

        Args:
            summary: Summary of the story
            target_genre: Target genre for styling
            style_examples: Style examples from the retriever

        Returns:
            Formatted prompt
        """
        return f"""
            You are a Narrative Style Transferer that converts story summaries into full narratives written in specific genre styles.

            # Input Summary:
            {summary}

            # Target Genre:
            {target_genre}

            {style_examples}

            # Task:
            Transform the input summary into a complete narrative written in the style of {target_genre} genre. 
            Pay close attention to the style examples provided and incorporate the characteristic elements of the genre.

            # Requirements:
            1. Maintain the core plot elements and characters from the summary
            2. Use vocabulary, sentence structure, and themes characteristic of {target_genre}
            3. Organize the story into coherent sections with appropriate titles
            4. Create an engaging title for the overall story
            5. Include a brief author's note explaining your stylistic approach

            # Output Format:
            Provide your response as a JSON object with the following structure:
            ```json
            {{
            "title": "Title of the styled story",
            "genre": "{target_genre}",
            "author_note": "Brief note about the style and approach",
            "sections": [
                {{
                "title": "Section Title",
                "content": "Section content..."
                }},
                {{
                "title": "Another Section",
                "content": "More content..."
                }}
            ],
            "style_elements": [
                {{
                "element_type": "vocabulary",
                "description": "Description of vocabulary choices",
                "examples": ["example1", "example2", "example3"]
                }},
                {{
                "element_type": "structure",
                "description": "Description of sentence/narrative structure",
                "examples": ["example1", "example2"]
                }}
            ]
            }}
            ```
        """

    def transform_with_style_function(
        self,
        summary: str,
        target_genre: str,
        style_intensity: float = 0.7,
        preserve_plot_elements: bool = True,
    ) -> Dict[str, Any]:
        """
        Function calling implementation for style transformation.

        Args:
            summary: Summary of the story
            target_genre: Target genre for styling
            style_intensity: How strongly to apply the style (0.0-1.0)
            preserve_plot_elements: Whether to strictly preserve plot elements

        Returns:
            Structured story output
        """
        # Define function for the model to call
        # Convert Pydantic models to function schema
        # Define a simpler schema that matches Google's expectations
        # Use FunctionDeclaration from the SDK
        transform_tool = FunctionDeclaration(
            name="transform_to_genre_style",
            description="Transform a story summary into a specific genre style",
            # Define parameters using the Schema object
            parameters=Schema(
                type=Type.OBJECT,
                properties={
                    "title": Schema(
                        type=Type.STRING, description="Title of the styled story"
                    ),
                    "genre": Schema(
                        type=Type.STRING, description="Genre/style of the story"
                    ),
                    "author_note": Schema(
                        type=Type.STRING,
                        description="Brief note about the style and approach",
                    ),
                    "sections": Schema(
                        type=Type.ARRAY,
                        description="Sections of the story",
                        items=Schema(
                            type=Type.OBJECT,
                            properties={
                                "title": Schema(
                                    type=Type.STRING,
                                    description="Title of this section",
                                ),
                                "content": Schema(
                                    type=Type.STRING,
                                    description="Content of this section",
                                ),
                            },
                            required=["title", "content"],
                        ),
                    ),
                    "style_elements": Schema(
                        type=Type.ARRAY,
                        description="Style elements used in the story",
                        items=Schema(
                            type=Type.OBJECT,
                            properties={
                                "element_type": Schema(
                                    type=Type.STRING,
                                    description="Type of style element",
                                ),
                                "description": Schema(
                                    type=Type.STRING,
                                    description="Description of the style element",
                                ),
                                "examples": Schema(
                                    type=Type.ARRAY,
                                    description="Examples of this style element",
                                    items=Schema(type=Type.STRING),
                                ),
                            },
                            required=["element_type", "description", "examples"],
                        ),
                    ),
                },
                required=[
                    "title",
                    "genre",
                    "author_note",
                    "sections",
                    "style_elements",
                ],
            ),
        )

        # Wrap the FunctionDeclaration in a Tool object
        tools = Tool(function_declarations=[transform_tool])

        # Prompt for function calling
        prompt = f"""
            You are a Narrative Style Transferer that converts story summaries into full narratives written in specific genre styles.

            # Input Summary:
            {summary}

            # Target Genre:
            {target_genre}

            # Style Intensity:
            {style_intensity} (0.0-1.0, where 1.0 is most intense)

            # Preserve Plot:
            {preserve_plot_elements} (whether to strictly maintain plot elements)

            # Task:
            Transform the input summary into a complete narrative written in the style of {target_genre} genre.

            Your response should be a complete story with the following elements:
            1. A title that captures the essence of the story
            2. An author's note explaining your stylistic approach
            3. Multiple sections with engaging content
            4. Style elements that highlight the techniques used

            Provide your response using the transform_to_genre_style function, which will structure your output with all these elements.
            Do not ask for additional information - generate all required content based on the input summary.
        """

        # Generate with function calling
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                candidate_count=1,
                max_output_tokens=4096,
            ),
            tools=tools,
        )

        # Extract function call result
        try:
            # Get the first candidate's content parts
            parts = response.candidates[0].content.parts

            # Get the first part which should contain the function call
            if parts and hasattr(parts[0], "function_call"):
                function_call = parts[0].function_call
                # Verify the function name
                if function_call.name == "transform_to_genre_style":
                    # The args are already a dict, no need for json.loads
                    args = function_call.args
                    # Validate with our model
                    validated_story = StyledStory(**args)
                    return validated_story.model_dump()
                else:
                    return {"error": f"Unexpected function call: {function_call.name}"}

            # If we get here, no function call was found
            return {"error": "No function call found in response"}
        except Exception as e:
            return {"error": f"Failed to parse response: {str(e)}"}
