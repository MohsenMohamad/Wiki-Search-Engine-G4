from flask import Flask, request, jsonify
from collections import Counter, defaultdict
import json, math, pickle, re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from google.cloud import storage
from inverted_index_gcp import InvertedIndex

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

apply_stemming = False
stemmer = None
if apply_stemming:
  stemmer = PorterStemmer()
bucket_name = 'final-project-full-corpus-tf-sorted-64500'
if apply_stemming:
    bucket_name = 'final-project-full-corpus-tf-sorted-64500-stem'
# Connect to bucket
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)

# Auxiliary function to read dictionary from bucket
def read_dict(dict_path):
  blob = bucket.blob(dict_path)
  contents = blob.download_as_bytes()
  return pickle.loads(contents)

nltk.download('stopwords')
stopwords_frozen = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
all_stopwords = stopwords_frozen.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

bin_dir = '.'
# Read title, body and anchor indexes from bucket
title_inverted_index = InvertedIndex.read_index('postings_gcp', 'title_index', bucket_name)
body_inverted_index = InvertedIndex.read_index('postings_gcp', 'body_index', bucket_name)
anchor_inverted_index = InvertedIndex.read_index('postings_gcp', 'anchor_index', bucket_name)
# Read dictionaries from bucket for doc_id as key and title, PageRank or page view as corresponding value
doc_titles = read_dict('postings_gcp/titles.pkl')
page_rank = read_dict('postings_gcp/pagerank.pkl')
page_view = read_dict('postings_gcp/pageview.pkl')
# Normalize all PageRank values using division by maximum PageRank value 
max_rank_value = max(page_rank.values())
norm_page_rank = {doc_id:(rank / max_rank_value) for doc_id, rank in page_rank.items()}
# Normalize all PageRank values using division by maximum PageRank value
max_view_value = max(page_view.values())
norm_page_view = {doc_id:(view / max_view_value) for doc_id, view in page_view.items()}

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    title_res_weight = 0.95
    body_res_weight = 1.3
    anchor_res_weight = 1.15
    pageview_res_weight = 1.5
    pagerank_res_weight = 0.8
    combined_score = Counter()
    tokenized_query = tokenize(query, apply_stemming)
    combined_score = combine_weighted_scores(combined_score, normalize_scores(calc_binary_score_title(tokenized_query, title_inverted_index, bin_dir)), title_res_weight)
    combined_score = combine_weighted_scores(combined_score, normalize_scores(calc_BM25_score(tokenized_query, body_inverted_index, bin_dir)), body_res_weight)
    combined_score = combine_weighted_scores(combined_score, normalize_scores(calc_binary_score_anchor(tokenized_query, anchor_inverted_index, bin_dir)), anchor_res_weight)
    for doc_id in combined_score.keys():
      combined_score[doc_id] += norm_page_rank.get(doc_id, 0) * pagerank_res_weight + norm_page_view.get(doc_id, 0) * pageview_res_weight
    res = [(str(doc_id), doc_titles.get(doc_id, "")) for doc_id, score in combined_score.most_common()][:100]
    if len(res) == 0:
      res = [(None,None)]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = tokenize(query, True)
    if len(res) == 0:
      res = [(None,None)]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


def tokenize(text, use_stemming=False):
  """
    This function tokenizes a text into a list of tokens. In addition, it filters stopwords and performs stemming.
    Parameters:
    -----------
    text: string, represting the text to tokenize.
    apply_stemming: boolean flag that notes whether to apply stemming
    Returns:
    -----------
    list of tokens.
  """
  tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
  if use_stemming:
    tokens = [stemmer.stem(word) for word in tokens]
  return tokens


# Auxiliary function to normalize a list of scores by division by max score in the list
def normalize_scores(doc_score_pairs):
  if (len(doc_score_pairs) == 0):
    return doc_score_pairs
  max_score = doc_score_pairs[0][1]
  return [(doc_id, (score / max_score)) for doc_id, score in doc_score_pairs]


# Auxiliary function to combine a list of scores for documents into a scores dictionary
def combine_weighted_scores(doc_scores_dict, scores_list, list_weight):
  for doc_id, score in scores_list:
      doc_scores_dict[doc_id] = doc_scores_dict.get(doc_id, 0) + score * list_weight
  return doc_scores_dict


# Calculates binary score for each token in the query by fetching all the documents where it appears.
def calc_binary_score_title(query_tokens, index, bin_dir_name, top_k=500):
  docs = {}
  for token in query_tokens:
    if token in index.df:
      try:
        pls = index.read_a_posting_list(bin_dir_name, token, bucket_name)
        for doc_id, tf in pls:
          docs[doc_id] = docs.get(doc_id, 0) + 1
      except:
        continue
  docs = [(doc_id, score) for doc_id, score in docs.items()]
  return sorted(docs, key=lambda x: x[1], reverse=True)[:top_k]


# Calculates binary score for each token in the query by fetching all the documents where it appears.
def calc_binary_score_anchor(query_tokens, index, bin_dir_name, top_k=500):
  query_to_search = list(set(query_tokens))
  query_length = len(query_to_search)
  k_length = round(top_k / query_length)
  docs = {}
  for token in query_to_search:
    if token in index.df:
      try:
        pls = index.read_a_posting_list(bin_dir_name, token, bucket_name)
        # taking only k_length documents with highest tf
        pls = pls[:k_length]
        for doc_id, tf in pls:
          docs[doc_id] = docs.get(doc_id, 0) + tf
      except:
        continue
  docs = [(doc_id, score) for doc_id, score in docs.items()]
  return sorted(docs, key=lambda x: x[1], reverse=True)[:top_k]


# Calculates BM25 score for each token in the query by fetching all the documents where it appears.
def calc_BM25_score(query_to_search, index, bin_dir_name, top_k=500, k1=1.2, b=0.75):
  query_to_search = list(set(query_to_search))
  query_length = len(query_to_search)
  k_length = round(top_k / query_length)
  doc_BM25_scores = Counter()
  for token in query_to_search:
    if token in index.df:
      # calculate idf for specific token as needed for OKAPI BM25 formula
      token_idf = math.log(1 + (len(index.dl) - index.df[token] + 0.5) / (index.df[token] + 0.5))
      pls = index.read_a_posting_list(bin_dir_name, token, bucket_name)
      # taking only k_length documents with highest tf
      pls = pls[:k_length]
      for doc_id, freq in pls:
        numerator = token_idf * freq * (k1 + 1)
        denominator = freq + k1 * (1 - b + (b * index.dl[doc_id]) / index.avg_dl)
        if (denominator != 0):
          doc_BM25_scores[doc_id] += (numerator / denominator)
  return [(doc_id, score) for doc_id, score in doc_BM25_scores.most_common()[:top_k]]


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    # app.run(host='0.0.0.0', port=8080, debug=True)
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)
