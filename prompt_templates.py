"""Prompt templates for the Gemini AI breed information service."""


def get_breed_info_prompt(breed_name: str) -> str:
    """Generate a structured prompt requesting key breed attributes from Gemini.

    Args:
        breed_name (str): Human-readable dog breed name (e.g., 'Golden Retriever').

    Returns:
        str: Formatted prompt string requesting breed information in strict Key: Value format.
    """
    return f"""You are a professional dog breed expert. Provide accurate, concise information about the dog breed: {breed_name}.

Return EXACTLY the following 10 fields, one per line, using this strict format:
Key: Value

Origin: [country or region of origin]
Lifespan: [typical lifespan range in years]
Weight: [typical weight range in kg or lbs]
Height: [typical height range in cm or inches]
Temperament: [3-5 key temperament traits, comma separated]
Exercise: [brief exercise needs, 1 sentence]
Diet: [brief dietary recommendation, 1 sentence]
Grooming: [brief grooming needs, 1 sentence]
Health Issues: [2-4 common health concerns, comma separated]
Interesting Fact: [one unique or surprising fact about this breed]

Do not include any extra text, headings, bullet points, or markdown formatting.
Only output the 10 lines in the exact format shown above."""
