import time
from pprint import pprint
from typing import List, Dict
from IRQA.qa import QuestionAnswering
from IRQA.ir import Query, DocumentRetrieval, PassageRetrieval

model_ckpt = 'distilbert-base-uncased-distilled-squad'

qa_model = QuestionAnswering(model_ckpt, model_ckpt)
start = time.time()
dr = DocumentRetrieval()
end = time.time()
print(f"DocumentRetrieval Creation took {end - start}")

def predict(question: str) -> List[Dict]:
    query = Query(question)
    start = time.time()
    documents = dr.retrieve(query)
    end = time.time()
    print(f"Document retrieval took {end - start}")
    start = time.time()
    pr = PassageRetrieval(documents)
    end = time.time()
    print(f"PassageRetrieval Creation took {end - start}")
    start = time.time()
    passages = pr.retrieve(query)
    end = time.time()
    print(f"Passage Retrieval took {end-start}")
    results = []
    for passage in passages:
        start = time.time()
        result = qa_model.answer(passage, query)
        end = time.time()
        print(f"QA took {end - start}")
        results.append(result)

    return sorted(results, key = lambda x: x['score'], reverse=True) 

if __name__ == "__main__":
    query = "Who is the mother of Solomon"
    pprint(predict(query))


