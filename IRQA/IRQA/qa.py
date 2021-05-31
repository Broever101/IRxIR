from IRQA.ir import Document, Query
from typing import Any
from transformers import AutoModelForQuestionAnswering
from transformers import AutoTokenizer
from transformers import QuestionAnsweringPipeline

class QuestionAnswering:
    def __init__(self, model_ckpt: Any=None, custom_tokenizer: Any=None) -> None:
        
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_ckpt)
        self.tokenizer = AutoTokenizer.from_pretrained(custom_tokenizer)
        self.pipeline = QuestionAnsweringPipeline(model=self.model, tokenizer=self.tokenizer, \
                                        framework='pt', device=0)

    def answer(self, passage: Document, query: Query) -> dict:
        result_set = {"book" : passage.book, "chapter" : passage.chapter_no, \
                        "context" : passage.text}
        result = self.pipeline(context= str(passage), question= query.text)
        result_set['answer'] = result['answer']
        result_set['score'] = result['score']
        return result_set