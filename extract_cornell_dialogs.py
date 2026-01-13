#!/usr/bin/env python3
"""
Extract dialog lines from Cornell Movie Dialog Corpus.

The Cornell corpus stores dialogs in movie_lines.txt with format:
L1045 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ They do not!

This script extracts just the dialog text (last field) and writes to dialog_corpus.txt
"""

import os

def extract_dialogs():
    """Extract dialog text from Cornell corpus."""
    corpus_dir = "cornell movie-dialogs corpus"
    lines_file = os.path.join(corpus_dir, "movie_lines.txt")
    output_file = "dialog_corpus.txt"

    if not os.path.exists(lines_file):
        print(f"Error: {lines_file} not found!")
        print(f"Make sure you've downloaded and unzipped the Cornell Movie Dialog Corpus.")
        return False

    print(f"Extracting dialogs from {lines_file}...")

    count = 0
    with open(lines_file, 'r', encoding='utf-8', errors='ignore') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                # Split on +++$+++ separator
                parts = line.strip().split(' +++$+++ ')
                if len(parts) >= 5:
                    # Last part is the actual dialog text
                    dialog_text = parts[-1]
                    f_out.write(dialog_text + '\n')
                    count += 1

    print(f"✓ Extracted {count:,} dialog lines to {output_file}")

    # Show file size
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✓ Output file size: {size_mb:.1f} MB")

    return True

if __name__ == "__main__":
    print("=" * 70)
    print("Cornell Movie Dialog Corpus - Dialog Extraction")
    print("=" * 70)
    print()

    success = extract_dialogs()

    if success:
        print()
        print("Next step: Build the unit catalog")
        print("  python3 prototype_topdown_units.py")
    else:
        print("\nPlease download the Cornell corpus first:")
        print("  wget http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip")
        print("  unzip cornell_movie_dialogs_corpus.zip")
