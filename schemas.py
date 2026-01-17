from pydantic import BaseModel
from typing import Literal

# This must match exactly what main.py is looking for
class PredictRequest(BaseModel):
    PClass: Literal["First", "Second", "Third"]
    Gender: Literal["Male", "Female"]
    Sibling: Literal["Zero", "One", "Two", "Three"]
    Embarked: Literal["Southampton", "Cherbourg", "Queenstown"]

class PredictResponse(BaseModel):
    predicted_label: int
    prediction: Literal["SURVIVED", "NOT SURVIVED"]