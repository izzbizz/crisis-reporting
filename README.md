# crisis-reporting

using graphical models, i compare global media reports on two different crises from 2019: cyclone idai in southeast africa and hurricane dorian in the caribbean, to answer the question: why does one event receive more media attention than another? and how is medial interest generated?

### steps:

1. list of news articles from the [gdelt](https://www.gdeltproject.org/) database
2. extraction of articles from urls, using [newspaper3k](https://github.com/codelucas/newspaper)
3. preprocessing: removal of duplicates, text statistics
4. first analysis: named entities (using [spacy](https://github.com/explosion/spaCy)) & websites
5. lda topic models with [gensim](https://github.com/RaRe-Technologies/gensim)
6. retrospective event detection as described in [wang et al.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.90.9651&rep=rep1&type=pdf)

### repo content:

- code: lda and red scripts
- notebooks: jupyter notebooks for pre- and post-processing of the corpora
