Building upon ideas found in the papers at https://nonlanguage.dev/causal-hnet/papers/ and those referenced by some of them and those referenced by those, etc...

## Documentation

[`docs/parser_doc.html`](docs/parser_doc.html) explains the energy relaxation parser algorithm, how it maps to
a neural feedforward/disinhibition timeline, and includes worked examples for "we should go" and
"I found my hat" with embedded raster plots generated from real corpus data.

To browse the documentation and interactive parse trees as a website, enable **GitHub Pages** on this
repository (Settings → Pages → Source: branch `experimental/substitution-parsing`, folder `/docs`).
The parse tree HTML files will then be live at `https://<username>.github.io/causal-hnet/`.

## Running

```bash
# Parse a phrase and view the neural timeline raster (requires unit_catalog.pkl)
python3 timeline_parse.py "we should go"
python3 timeline_parse.py "i found my hat"

# Generate an interactive parse tree HTML
python3 analyze_phrase.py "we should go"

# View the raster display with demo data (no catalog needed)
python3 neural_demo.py
python3 neural_demo.py --phrase 4w
```
