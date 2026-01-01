from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from .models import User

def index(request):
    users = User.objects.all()
    data = ", ".join([u.name for u in users])
    return HttpResponse(f"Users: {data}")

