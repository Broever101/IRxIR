from pprint import pprint 
from retrieval.model import predict 

if __name__ == "__main__":
    query = "Who is Judah"
    pprint(predict(query))