# Word2Vec from Scratch on 20 Newsgroups

A skip-gram Word2Vec model with negative sampling, implemented in PyTorch and trained on the [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/) dataset (~18,000 forum posts across 20 topics).

The model learns a 100-dimensional vector for each word, so that words used in similar contexts end up close together:

| Query | Nearest neighbours |
|---|---|
| `god` | righteousness, deity, omnipotent, spirit, creator |
| `car` | cars, tire, dealership, truck, toyota |
| `windows` | workgroups, apps, microsoft, excel, desktop |
| `space` | shuttle, hubble, payloads, mir |

## How it works

All of the code is in [`project.ipynb`](project.ipynb).

1. **Preprocessing.** Load the posts with headers, footers and quotes removed. Lowercase the text, keep only letters, and drop stopwords and tokens shorter than 3 characters.
2. **Vocabulary.** Keep words that appear at least 4 times (~30k words), and give each word an index in alphabetical order.
3. **Training pairs.** Slide a window of ±2 words over each post to produce (target, context) pairs (~7.8M).
4. **Model.** Two embedding tables, one for target words and one for context words. For each real pair, the model draws 5 negative words from the unigram distribution raised to the power 0.75. It is trained with a logistic loss that scores the real pair high and the negatives low.
5. **Evaluation.** Find a word's nearest neighbours by cosine similarity between the normalised target embeddings.

| Hyperparameter | Value |
|---|---|
| Embedding size | 100 |
| Context window | 2 |
| Negative samples | 5 |
| Batch size | 1024 |
| Optimizer | Adam, lr = 1e-3 |
| Epochs | 3 |

Training loss goes from ~4.16 (6 × ln 2 at initialisation) to 1.95 after 3 epochs. It runs on Apple Silicon GPUs through `mps`, and falls back to the CPU otherwise.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch scikit-learn nltk jupyter
jupyter notebook project.ipynb
```

The dataset and the NLTK stopword list download automatically the first time you run the notebook.
