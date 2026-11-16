"""Serve the trained shopping session classifier with FastAPI."""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent
EXAMPLE = json.loads((ROOT / "examples" / "session.json").read_text())
Count = Annotated[int, Field(ge=0)]
Duration = Annotated[float, Field(ge=0)]
Code = Annotated[int, Field(ge=1)]


class Session(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False,
        json_schema_extra={"examples": [EXAMPLE]},
    )

    Administrative: Count
    Administrative_Duration: Duration
    Informational: Count
    Informational_Duration: Duration
    ProductRelated: Count
    ProductRelated_Duration: Duration
    SpecialDay: Annotated[float, Field(ge=0, le=1)]
    Month: Literal["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    OperatingSystems: Code
    Browser: Code
    Region: Code
    TrafficType: Code
    VisitorType: Literal["New_Visitor", "Returning_Visitor", "Other"]
    Weekend: bool


class Prediction(BaseModel):
    purchase_probability: float
    model: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.environ.get("MODEL_PATH", ROOT / "artifacts" / "model.joblib"))
    app.state.model = joblib.load(path)
    yield


app = FastAPI(title="Shopping Purchase Prediction", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse("/docs")


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model": request.app.state.model["name"]}


@app.post("/predict", response_model=Prediction)
def predict(session: Session, request: Request):
    bundle = request.app.state.model
    features = pd.DataFrame([session.model_dump()])
    score = bundle["pipeline"].predict_proba(features)[0, 1]
    return Prediction(purchase_probability=float(score), model=bundle["name"])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "9696")))
