import re

def count_brand_mentions(text: str, brand: str) -> int:
    pattern = re.compile(re.escape(brand), re.IGNORECASE)
    return len(pattern.findall(text))
