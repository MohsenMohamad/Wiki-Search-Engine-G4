# Wiki-Search-Engine-G4

# Authors:
Mohamad Mohsen, Alexander Trachtenberg

# Final project in Information Retrieval course at Ben Gurion University

In this project we bulit a search engine to the entire English Wikipedia corpus, above 6,300,000 documents (!).

We used all course's theory in order to develop our search engine:

1. # Preprocessing (build_inverted_index_gcp.ipynb):
   Each Wiki page was preprocessed by seperating its different sections into three main components:
   Title, Body text and Anchor text.
   Then we parsed, tokenized, removed stopwords and rare words, and used stemming if needed.

2. # Inverted index (build_inverted_index_gcp.ipynb):
   We've built three inverted indexes for the three main components: Title, Body text and Anchor text.
   Only the inverted index for the body component contains a dictionary of doc_id and its length.

3. # Search throught the data (search_frontend.py):
   After preprocessing the data we used a similarity function to determine the similarity between a given query to a list of candidate documents, using the inverted index.
   Okapi BM25 bag-of-words similarity function was used for the body component.
   Boolean similarity function was used for the title and anchor components.
   
4. # Ranking (search_frontend.py):
   Documents scores, calculated from similarity function of the given query and documents from the title, body and anchor components, were normalized by division by max score in each component.
   Also, PageRank and page view amount for each document were normalized and combined into a ranking function with five components: title, body, anchor, PageRank and page view. When combining the 
   scores, each component was multiplied by its weight. Those weights were adjusted to get optimal ranking results.
      
6. # Evaluation
   The course staff provided us with labled benchmark (queries_train.json) as train dataset for tesing and evaluation. The evaluation metrics that were used:
   Precision@5: Number of relevant documents for a given query that were retrieved, divided by the total number of retrived documents. In our case we looked at first 5 documents retrieved.
   F1@30: Harmonic mean of the precision and recall metrics. In our case we calculated the precision and recall of first 30 documents retrieved.
