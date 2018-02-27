# LanguageModels

# Files
- generate_sentence.py implements bi-gram kneser ney language model.
- Modified_KN implements modified kneser ney model (Chen and Goodman https://people.eecs.berkeley.edu/~klein/cs294-5/chen_goodman.pdf). The corner cases where both the ngram and interpolation term becomes zero (or divide by zero) aren't handled well.

# Usage
- Environment : Anaconda + nltk + tqdm(not needed for generate_sentence.py )
- unzip data
- run *python generate_sentence.py* to generate a 10 token sentence

# Datasets used:
Brown and Gutenberg corpus



