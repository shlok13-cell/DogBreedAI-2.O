"""Gemini AI integration service for fetching detailed dog breed information and interactive chat."""

import os
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

from prompt_templates import get_breed_info_prompt

# Load environment variables from .env file at project root
load_dotenv()


def get_gemini_api_key() -> Optional[str]:
    """Retrieve GEMINI_API_KEY from environment variables.

    Returns:
        Optional[str]: API key string if present and configured, else None.
    """
    key = os.getenv("GEMINI_API_KEY")
    if not key or not key.strip() or "your_gemini_api_key" in key.lower():
        return None
    return key.strip()


def get_breed_info(breed_name: str) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """Query Google Gemini AI to retrieve structured dog breed information.

    Args:
        breed_name (str): Formatted breed title (e.g., 'Golden Retriever').

    Returns:
        Tuple[Optional[Dict[str, str]], Optional[str]]:
            - Dictionary mapping attribute titles (Origin, Lifespan, etc.) to values.
            - Error or warning string if key is missing or request fails.
    """
    api_key = get_gemini_api_key()
    if not api_key:
        return (
            None,
            "🔑 **GEMINI_API_KEY Not Configured**: Please set your API key in the `.env` file to enable AI breed insights.",
        )

    prompt = get_breed_info_prompt(breed_name)
    candidate_models = [
        "gemini-flash-latest",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-pro-latest",
    ]

    last_error: Optional[str] = None

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        for model_name in candidate_models:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                response_text = response.text

                if not response_text:
                    continue

                # Parse strict Key: Value lines
                breed_info: Dict[str, str] = {}
                for line in response_text.strip().split("\n"):
                    line = line.strip()
                    if ":" in line:
                        key, val = line.split(":", 1)
                        clean_key = key.strip().strip("*#_-\"` ")
                        clean_val = val.strip().strip("*#_-\"` ")
                        if clean_key and clean_val:
                            breed_info[clean_key] = clean_val

                if breed_info:
                    return breed_info, None

            except Exception as e:
                err_str = str(e)
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                    last_error = "⏳ **Gemini API Quota Exceeded**: Free tier request limit reached. Please wait a minute or check your Google AI Studio account billing."
                else:
                    last_error = f"Gemini API error ({model_name}): {err_str}"

    except Exception as outer_err:
        last_error = f"Failed to initialize Gemini AI client: {outer_err}"

    return None, last_error or "Unable to retrieve breed information from Gemini API."


def chat_with_breed_expert(
    breed_name: str,
    conversation_history: List[Dict[str, str]],
    user_message: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Conduct a multi-turn chat session grounded in the predicted dog breed context.

    Args:
        breed_name (str): The predicted dog breed name (e.g., 'Golden Retriever').
        conversation_history (List[Dict[str, str]]): List of previous chat dicts with 'role' ('user'/'assistant') and 'content'.
        user_message (str): New user message text.

    Returns:
        Tuple[Optional[str], Optional[str]]:
            - AI assistant response text.
            - Error or warning message if key is missing or request fails.
    """
    api_key = get_gemini_api_key()
    if not api_key:
        return (
            None,
            "🔑 **GEMINI_API_KEY Not Configured**: Please set your API key in `.env` to chat with the AI assistant.",
        )

    system_instruction = (
        f"You are an expert canine veterinarian and dog breed specialist. "
        f"The user is currently analyzing the predicted dog breed: **{breed_name}**.\n\n"
        f"CRITICAL INSTRUCTIONS YOU MUST FOLLOW:\n"
        f"1. Answer ALL user questions specifically in the context of the breed **{breed_name}** unless the user explicitly asks to compare it with another specific dog breed.\n"
        f"2. You MUST ONLY answer questions related to dogs, dog breeds, care, training, health, behavior, diet, grooming, and characteristics.\n"
        f"3. If the user asks a question that is completely unrelated to dogs or dog breeds (such as math, coding, politics, recipes, or movies), politely decline with this response: "
        f"'I am your Dog Breed AI Assistant! I can only answer questions related to dog breeds, care, training, and health. Please ask me anything about {breed_name} or other dog breeds!'\n"
        f"4. Keep your responses clear, friendly, well-formatted, and concise."
    )

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7,
        )

        # Build contents list preserving history
        contents = []
        for msg in conversation_history:
            role_tag = "user" if msg.get("role") == "user" else "model"
            msg_text = msg.get("content", "")
            if msg_text:
                contents.append(
                    types.Content(
                        role=role_tag,
                        parts=[types.Part.from_text(text=msg_text)],
                    )
                )

        # Append current user question
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_message)],
            )
        )

        candidate_models = ["gemini-flash-latest", "gemini-2.0-flash", "gemini-2.0-flash-lite"]
        last_error = None

        for model_name in candidate_models:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                if response and response.text:
                    return response.text.strip(), None
            except Exception as e:
                err_str = str(e)
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                    last_error = "⏳ **Gemini API Quota Exceeded**: Request limit reached. Please wait a minute before sending another message."
                else:
                    last_error = f"Gemini chat error ({model_name}): {err_str}"

        return None, last_error or "Unable to generate AI chat response."

    except Exception as outer_err:
        return None, f"Failed to initialize Gemini AI chat client: {outer_err}"
