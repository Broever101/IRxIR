from typing import List, Dict
from qa import QuestionAnswering
from ir import Query, DocumentRetrieval, PassageRetrieval

model_ckpt = 'distilbert-base-uncased-distilled-squad'

qa_model = QuestionAnswering(model_ckpt, model_ckpt)
dr = DocumentRetrieval()

def predict(question: str) -> List[Dict]:
    query = Query(question)
    documents = dr.retrieve(query)
    pr = PassageRetrieval(documents)
    passages = pr.retrieve(query)
    results = []
    for passage in passages:
        result = qa_model.answer(passage, query)
        results.append(result)
    return results 