def clean_tags(tag_value):
    """
    Cleans the Tags column entry.
    - Splits by comma
    - Strips whitespace
    - Handles NaN / empty cases
    Returns: list of tags
    """

    if not str(tag_value).strip():
        return []
    return [tag.strip() for tag in str(tag_value).split(",")]