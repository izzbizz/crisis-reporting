# Jupyter notebooks

These notebooks are examples of the pre- and post-processing (plotting and evaluating the lda results) of the corpora.

### Pre-processing

- Scrape articles from web using GDELT's URLs
- Drop duplicates
- Drop empty articles
- Drop article if no keywords occur

### First analysis

- Turn SQL-date into datetime object
- Count number of words
- Get quotes with regexes
- Plot article count per day
- Plot quoted ratio per day
- Plot text length per day
- Get named entities with spacy
- Plot most common sources by continent, number of articles

### Post-processing

- Plot LDA topics by day
- Compute and plot Hellinger distance between days
