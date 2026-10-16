# Licences (checked 2026-10-16)

| Component | Licence | Where we checked |
|---|---|---|
| Our code, README, results text (this repository) | MIT, see `LICENSE` | - |
| Model `google/canine-s` (CANINE-s, about 0.13B parameters; the Hugging Face card shows 0.1B, F32 weights 528 MB) | **Apache-2.0** | Model card metadata `license: apache-2.0`: https://huggingface.co/google/canine-s (revision `75d6d0b3f4d0bffed0ca7ddd0b73949977d4ca02`). Code: https://github.com/google-research/language/tree/master/language/canine |
| Data: Vietnamese Wikipedia, dump 2023-11-01 as published by Wikimedia on Hugging Face | **CC BY-SA 3.0 and GFDL** (text) | Dataset card, "Licensing information": https://huggingface.co/datasets/wikimedia/wikipedia (revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`). Wikimedia dump licence page, which today says CC BY-SA 4.0 and GFDL: https://dumps.wikimedia.org/legal.html . Terms of Use: https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use |
| `torch`, `transformers`, `pyarrow`, `numpy`, `huggingface_hub` | BSD-3-Clause / Apache-2.0 / Apache-2.0 / BSD-3-Clause / Apache-2.0 | Their PyPI pages; installed from PyPI, not redistributed here |

## What this means for this repository

- The Wikipedia text is **share-alike with attribution**. We do **not** redistribute the corpus or any trained weights.
  The only Wikipedia text in the repo is a few dozen short test sentences quoted in `results.md` and `results/examples.json`,
  each with its article title; those excerpts stay under CC BY-SA 4.0 / GFDL (authors: see each article's history on vi.wikipedia.org),
  not under our MIT licence. Everything else (code, numbers, prose) is MIT.
- The metrics and the cost figures are our own measurements and carry no data licence.
- If you retrain and publish weights: the Apache-2.0 base model and a share-alike corpus mean you should read the CC BY-SA terms
  before publishing weights trained on this data. We have not taken legal advice on whether weights are derivative works, and publish none.
- We collected no personal data. Wikipedia articles can mention public persons; we kept only sentences made of letters, digits and basic punctuation
  and did **not** scan for personal data, so named public figures can appear in sentences. Nothing is attached to them beyond the public article text.

## Not used, and why (Arabic)

The dataset that fits the usual Arabic diacritization request, Tashkeela, is **GPL-2.0** (SourceForge project page,
https://sourceforge.net/projects/tashkeela/ , and the Hugging Face copy `community-datasets/tashkeela`). Copies on the Hub that are labelled MIT or
Apache-2.0 appear to be re-uploads of Tashkeela and other sources; we did not verify their provenance, so we did not use them. We chose a corpus whose
licence we could read at the source. Nothing in this repo has been run on Arabic.
