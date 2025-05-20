from fastapi import FastAPI, Request
from typing import Dict
import pandas as pd
from NeuMF.inference import NeuralMatrixFactoration

app = FastAPI()

neumf = NeuralMatrixFactoration(
    weight_path="../weights/new_NeuMF.weights.h5",
    interaction_file="../data/interaction.csv",
    book_file="../data/final_books.csv"
)

@app.get("/")
def read_root():
    return {"Hello": "World"}



