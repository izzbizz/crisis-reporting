# crisis-reporting

Using graphical models, I compare global media reports on two different crises from 2019: Cyclone Idai in southeast Africa and Hurricane Dorian in the Caribbean, to answer the question: Why does one event receive more media attention than another? And how is medial interest generated?

### Steps:

1. List of news articles from the [GDELT](https://www.gdeltproject.org/) database
2. Extraction of articles from URLs, using [newspaper3k](https://github.com/codelucas/newspaper)
3. Preprocessing: removal of duplicates, text statistics
4. First analysis: named entities (using [spacy](https://github.com/explosion/spaCy)) & websites
5. LDA topic models with [gensim](https://github.com/RaRe-Technologies/gensim)
6. Retrospective event detection as described in [Li et al.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.90.9651&rep=rep1&type=pdf)

### Repo content:

- code: LDA and RED scripts
- code/notebooks: Jupyter notebooks for pre- and post-processing of the corpora
- data: raw CSVs from GDELT
