from typing import List, Dict
import spacy
import re

def clean_job_titles(title: str) -> List[Dict[str, str]]:
    """
    Clean job titles using spaCy NLP and regex patterns.

    Args:
        title: string

    Returns:
        Clean title string
    """

    # Define constants
    SENIORITY_TERMS = {
        'senior', 'sr', 'staff', 'principal', 'director',
        'associate', 'graduate', 'entry', 'junior', 'jr',
        'intern', 'trainee', 'chief', 'head', 'vice', 'assistant', 'dir',
        'vp', 'manager'
    }

    COMMON_COMPANIES = {
        'tiktok', 'google', 'microsoft', 'amazon', 'meta', 'apple',
        'facebook', 'netflix', 'uber', 'airbnb', 'tesla', 'nvidia'
    }

    nlp = spacy.load("en_core_web_sm")

    original_title = title.strip()
    cleaned = original_title.lower()
    seniority = ''

    # 1. Remove dates and years
    date_patterns = [
        r'- \d{4} Start',
        r'\d{4} Start',
        r'(Spring|Summer|Fall|Winter)\s+\d{4}',
        r'20\d{2}(?!\w)',
        r'- \d{4}(?!\w)'
    ]
    for pattern in date_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    # 2. Remove content in parentheses (locations, technical details, degrees)
    parentheses_content = re.findall(r'\([^)]+\)', cleaned)
    for content in parentheses_content:
        # Keep if it seems like part of the core job title
        content_clean = content.lower().strip('()')
        if not any(indicator in content_clean for indicator in
                  ['singapore', 'india', 'remote', 'platform', 'infrastructure',
                   'team', 'ms', 'phd', 'mba', 'level', 'speaking', 'h/f']):
            continue
        cleaned = cleaned.replace(content, '')

    # 3. Use spaCy to analyze the remaining text
    doc = nlp(cleaned)

    # Extract tokens and their properties
    tokens_to_keep = []

    for i, token in enumerate(doc):
        # Skip seniority terms
        if token.text.strip() in SENIORITY_TERMS:
            seniority = token.text.strip()
            continue

        # Skip company names at the beginning (before dash/colon)
        if (i < len(doc) - 2 and
            token.text.strip() in COMMON_COMPANIES and
            i < 3):  # Only at the beginning
            continue

        # Skip locations
        if token.ent_type_ in ['GPE', 'LOC']:  # Geopolitical entity, Location
            continue

        # Skip common stop words that are not meaningful
        if token in ['h', 'f', 'm'] or token.text in ['-', '/', ':', ',']:
            continue

        # Keep meaningful tokens
        if (not token.is_punct or token.text in ['+', '#', '.']) and len(token.text) > 1:
            tokens_to_keep.append(token.text)

    # 4. Additional regex cleaning for specific patterns
    rejoined = ' '.join(tokens_to_keep)

    # Remove company prefixes before dash/colon
    company_prefix_pattern = r'^[A-Za-z0-9\s&.,-]+\s*[-:]\s*'
    match = re.match(company_prefix_pattern, rejoined)
    if match and len(match.group(0)) < len(rejoined) * 0.4:  # Don't remove if it's most of the title
        rejoined = re.sub(company_prefix_pattern, '', rejoined)

    # Remove level pattern
    patterns = [
        r"\bL[1-4]\b",         # L1, L2
        r"Level\s?1",          # Level 1, Level1
        r"level",
        r"L-\d",               # L-1, L-2
        r"\bI\b",
        r"\bIII\b",            # Roman numeral III
        r"\bII\b",
        r"\bIV\b",
        r"class"
    ]

    # Remove matched patterns
    for pattern in patterns:
        rejoined = re.sub(pattern, "", rejoined, flags=re.IGNORECASE)

    # Remove language requirements
    rejoined = re.sub(r'[A-Za-z]+\s+Speaking', '', rejoined, flags=re.IGNORECASE)

    # Remove part/full time requirements
    rejoined = re.sub(r'[A-Za-z]+\s+time', '', rejoined, flags=re.IGNORECASE)

    # Remove trailing numbers and location patterns
    rejoined = re.sub(r'[A-Za-z]+-\d+$', '', rejoined)

    # Clean up spacing and punctuation
    rejoined = re.sub(r'\s*[-:,]\s*$', '', rejoined)  # Trailing punctuation
    rejoined = re.sub(r'^\s*[-:,]\s*', '', rejoined)   # Leading punctuation
    rejoined = re.sub(r'\s{2,}', ' ', rejoined)        # Multiple spaces
    rejoined = rejoined.strip()

    # 5. Handle edge cases
    if len(rejoined.split()) == 1 or not rejoined:
        rejoined = original_title  # Return original if too much was removed

    return rejoined