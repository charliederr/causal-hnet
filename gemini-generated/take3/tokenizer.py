import re

class SimpleTokenizer:
    def tokenize(self, text):
        # 1. Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # 2. Pad punctuation with spaces so they become tokens
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        # 3. Lowercase and split
        return text.lower().split()
