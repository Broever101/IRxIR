import os
import re
from typing import Any, Dict, List 
from pandas.core.frame import DataFrame
import cupy as np
import json
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

THRESHOLD = 200
STEMMER = PorterStemmer()

class Document:
    def __init__(self, book: str, chapter_no: int, text: List[str], ps: PorterStemmer= STEMMER) -> None:
      self.book = book
      self.chapter_no = chapter_no
      self.text = text
      self.ps = ps
      self.stemmed = self.stem()

    def stem(self) -> str:
      text = []
      for line in self.text:
        newline = ''
        for word in line:
          newline += self.ps.stem(word)
        text.append(newline)

      return text

    def __str__(self) -> str:
      return " ".join(self.text)

class Query:
    def __init__(self, text) -> None:
        #plain text
        self.text = text

    def __str__(self) -> str:
      return self.text

Corpus = List[Document]


def read_corpus(path: os.path, name: str, ps: PorterStemmer= STEMMER) -> Corpus:
    with open(path) as f:
        bible = json.load(f)

    def preprocess(text):
        p = re.compile("<(/)?\w>")
        text = p.sub("", text)
        text = text.replace("\n", " ")
        return text.lower()

    book = bible[name]
    chapters = list(book.keys())
    corpus = []
    for i, chapter in enumerate(chapters):
        text = book[chapter][1:]
        text = ["\n".join(passage[1:]) for passage in text]
        text = "\n".join(text)
        text = preprocess(text)
        text = text.split(".")
        for line in text:
            line += "."
        corpus.append(Document(chapter, i + 1, text, ps))
    return corpus


class DocumentRetrieval:
    def __init__(self, documents: Corpus=None, stemmer: PorterStemmer= STEMMER) -> None:
      self.ps = stemmer
      
      if not documents:
        path = os.path.join("data", "kjv.json")
        name = "KingJamesVersion"
        self.documents = read_corpus(path, name)
      else:
        self.documents = documents

      self.vectorizer = TfidfVectorizer(stop_words='english',
                                        binary = True,
                                        ngram_range=(1, 2),
                                        sublinear_tf=True)
      self.inverted_doc = self.corpus_vectorizer()
      self.vocab_len = self.inverted_doc.shape[0]
      self.doc_len = len(self.documents)

    def corpus_vectorizer(self) -> DataFrame: 
      docs = [" ".join(doc.stemmed) for doc in self.documents]
      X = self.vectorizer.fit_transform(docs)
      X = X.T.toarray()
      df = pd.DataFrame(X, index=self.vectorizer.get_feature_names())
      return df

    def query_vectorizer(self, query: Query) -> Any:
      q = [str(query)]
      q_vec = self.vectorizer.transform(q).toarray().reshape(self.vocab_len,)
      return q_vec


    def retrieve(self, query: Query, topk: int= 3, quiet: bool= True) -> Corpus:
      q_vec = self.query_vectorizer(query)
      
      sim = {}
      for i in range(self.doc_len):
          sim[i] = (self.inverted_doc.loc[:,i].dot(q_vec) \
              / np.linalg.norm(self.inverted_doc.loc[:, i])) \
              * np.linalg.norm(q_vec)

      sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)

      candidates = []
      for i, vals in enumerate(sim_sorted):
          if i >= topk:
            break
          
          k, v = vals
          if not quiet:
            print(f'Similarity: {v}')
            print(f'With Document: {doc.book}, {doc.chapter_no}')
          
          doc = self.documents[k]
          candidates.append(doc)

      return candidates


class PassageRetrieval(DocumentRetrieval):
  def __init__(self, documents: Corpus) -> None:
    docs = self.get_passages(documents)
    super().__init__(docs)

  def get_passages(self, documents: Corpus) -> Corpus:
    
    def chunks(doc: Document, threshold: int) -> str:
      text = str(doc)
      i = 0
      words = text.split(' ')
      while i < len(words):
        yield " ".join(words[i:i + threshold])
        i = i + threshold
      yield " ".join(words[i:])

    docs = []
    
    for doc in documents:
      for chunk in chunks(doc, THRESHOLD):
        if chunk not in ["", " "]:
          text = [x + '.' for x in chunk.split('.')]
          docs.append(Document(doc.book, doc.chapter_no, text))
    
    return docs 

  