#!/usr/bin/env python3
"""
Benchmark: Substitution-based prediction vs Transformer (GPT-2)

Compares next-word predictions from both systems.
"""

import time
import sys

# Import ContextPattern FIRST to fix pickle loading
from bidir_simple import UnitCatalog, SimpleBidirParser, ContextPattern

print("=" * 70)
print("BENCHMARK: Substitution Predictor vs GPT-2")
print("=" * 70)

# Check for transformers library
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch
    HAS_TRANSFORMERS = True
    print("GPT-2 model available via HuggingFace transformers")
except ImportError:
    HAS_TRANSFORMERS = False
    print("ERROR: transformers library not found. Install with:")
    print("  pip install transformers")
    sys.exit(1)

# Load GPT-2
print("\nLoading GPT-2...")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_model.eval()

# Use MPS if available
if torch.backends.mps.is_available():
    gpt2_device = torch.device("mps")
    gpt2_model = gpt2_model.to(gpt2_device)
    print(f"GPT-2 running on: MPS (Apple Metal)")
elif torch.cuda.is_available():
    gpt2_device = torch.device("cuda")
    gpt2_model = gpt2_model.to(gpt2_device)
    print(f"GPT-2 running on: CUDA")
else:
    gpt2_device = torch.device("cpu")
    print(f"GPT-2 running on: CPU")

# Load substitution predictor
print("\nLoading substitution predictor...")
from substitution_predictor import SubstitutionPredictor

catalog = UnitCatalog()
try:
    catalog.load('unit_catalog.pkl')
except FileNotFoundError:
    print("Error: unit_catalog.pkl not found. Run bidir_simple.py first.")
    sys.exit(1)

# Build GPU index
print("Building GPU index...")
catalog.build_gpu_index()

# Create predictor and parser
predictor = SubstitutionPredictor(catalog)
parser = SimpleBidirParser(catalog, debug=False)


def get_gpt2_predictions(prompt_text: str, top_k: int = 10):
    """Get top-k next word predictions from GPT-2."""
    inputs = gpt2_tokenizer(prompt_text, return_tensors="pt").to(gpt2_device)

    with torch.no_grad():
        outputs = gpt2_model(**inputs)
        # Get logits for the last token position
        next_token_logits = outputs.logits[0, -1, :]

        # Get top-k tokens
        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
        probs = torch.softmax(top_k_logits, dim=0)

        predictions = []
        for i in range(top_k):
            token = gpt2_tokenizer.decode(top_k_indices[i])
            prob = probs[i].item()
            predictions.append((token.strip(), prob))

    return predictions


def get_substitution_predictions(prompt_tokens: list, top_k: int = 10):
    """Get top-k predictions from substitution predictor."""
    try:
        preds = predictor.predict_with_substitution_class(
            prompt_tokens, parser, top_k=top_k
        )
        # UnitPrediction objects have .unit_text and .score attributes
        return [(p.unit_text, p.score) for p in preds]
    except Exception as e:
        print(f"  Substitution predictor error: {e}")
        return []


# Test prompts
test_prompts = [
    "I want to",
    "She said that",
    "We should go",
    "The weather is",
    "He didn't know",
    "They were going to",
    "It was a beautiful",
    "Can you help me",
]

print("\n" + "=" * 70)
print("NEXT-WORD PREDICTION COMPARISON")
print("=" * 70)

for prompt_text in test_prompts:
    prompt_tokens = prompt_text.lower().split()

    print(f"\nPROMPT: \"{prompt_text}\"")
    print("-" * 50)

    # GPT-2 predictions
    t0 = time.time()
    gpt2_preds = get_gpt2_predictions(prompt_text, top_k=10)
    gpt2_time = time.time() - t0

    print(f"\nGPT-2 predictions ({gpt2_time:.3f}s):")
    for i, (word, prob) in enumerate(gpt2_preds[:5], 1):
        print(f"  {i}. \"{word}\" ({prob:.3f})")

    # Substitution predictions
    t0 = time.time()
    sub_preds = get_substitution_predictions(prompt_tokens, top_k=10)
    sub_time = time.time() - t0

    print(f"\nSubstitution predictions ({sub_time:.3f}s):")
    if sub_preds:
        for i, (word, score) in enumerate(sub_preds[:5], 1):
            print(f"  {i}. \"{word}\" ({score:.3f})")
    else:
        print("  (no predictions)")

    # Check overlap
    gpt2_words = set(w.lower().strip() for w, _ in gpt2_preds[:10])
    sub_words = set(w.lower().strip() for w, _ in sub_preds[:10]) if sub_preds else set()
    overlap = gpt2_words & sub_words

    print(f"\nOverlap in top-10: {len(overlap)} words")
    if overlap:
        print(f"  Common: {overlap}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Key differences:
- GPT-2: Trained on WebText, predicts subword tokens
- Substitution: Uses dialog corpus patterns, predicts multi-word phrases

The systems have different strengths:
- GPT-2: General knowledge, fluent generation
- Substitution: Captures dialog-specific patterns and substitutability
""")
