from bs4 import BeautifulSoup
import html2text

def html_to_markdown(html: str) -> str:
    """
    Converts raw HTML content into clean, structured Markdown suitable for semantic search.

    This function:
    - Removes non-content tags such as <script>, <style>, <meta>, <footer>, etc.
    - Strips out empty <p> and <div> blocks
    - Converts remaining content to Markdown format
    - Preserves headings, lists, bold/italic text, and links

    Args:
        html (str): Raw HTML string to be cleaned and converted.

    Returns:
        str: Cleaned Markdown-formatted text.
    """
    
    # Step 1: Parse and clean HTML with BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted tags
    for tag in soup(["script", "style", "meta", "footer", "nav", "form", "iframe", "noscript"]):
        tag.decompose()

    # Remove empty <p> or <div> tags
    for tag in soup.find_all(["p", "div"]):
        if not tag.get_text(strip=True):
            tag.decompose()

    # Step 2: Convert to Markdown using html2text
    converter = html2text.HTML2Text()
    converter.ignore_links = False
    converter.ignore_images = True
    converter.ignore_emphasis = False
    converter.body_width = 0  # prevent line wrapping

    markdown = converter.handle(str(soup))
    return markdown.strip()
