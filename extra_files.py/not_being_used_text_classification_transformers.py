from transformers import pipeline

def classify_text(text: str) -> dict:
    """
    Classify text into categories using zero-shot classification.

    Args:
        text (str): The text to classify.
        categories (list[str]): The categories to classify the text into.

    Returns:
        dict: A dictionary containing the classification results.
    """
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    expertise = classifier(text, ["Tech", "Marketing", "Finance"], multi_label=False)
    role_level = classifier(text, ["Entry", "Mid", "Senior", "Executive"])
    org_unit = classifier(text, ["Engineering", "Sales", "HR"])

    return {
        "expertise": expertise,
        "role_level": role_level,
        "org_unit": org_unit
    }
