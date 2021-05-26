from qa import QuestionAnswering
from ir import Query, DocumentRetrieval, PassageRetrieval

model_ckpt = 'ktrapeznikov/albert-xlarge-v2-squad-v2'

qa_model = QuestionAnswering(model_ckpt, model_ckpt)
dr = DocumentRetrieval()

def predict(question: str) -> list[dict]:
    query = Query(question)
    documents = dr.retrieve(query)
    pr = PassageRetrieval(documents)
    passages = pr.retrieve(query)
    results = []
    for passage in passages:
        result = qa_model.answer(str(passage), query.text)
        results.append(result)
    return results 