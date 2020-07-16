# jupyter notebooks

these notebooks are examples of the pre- and post-processing (plotting and evaluating the lda results) that i did with the corpora.

### pre-processing

- scrape articles from web using gdelt URLs
- drop duplicates
- drop empty articles
- drop article if no keywords occur

### first analysis

- turn SQL-date into datetime object
- count number of words
- get quotes with regexes
- plot article count per day
- plot quoted ratio per day
- plot text length per day
- get named entities with spacy
- plot most common sources by continent, number of articles

### post-processing

- plot lda topics by day
- compute and plot hellinger distance between days
