from django.shortcuts import render
from IRQA.model import predict

def home(request):
    return render(request, "index.html")

def result(request):
    query = request.GET["query"]
    result = predict(query)
    return render(request, 'result.html', {'result':result})