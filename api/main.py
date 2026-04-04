"""
main.py — FastAPI inference service for phishing URL detection.

Usage:
    uvicorn main:app --reload --port 8000

Endpoints:
    POST /predict   { "url": "https://..." }  →  classification + explanation
    GET  /health    →  service status
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from predictor import Predictor


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title='Phishing URL Detection API',
    description='Classifies URLs as phishing or legitimate with SHAP-based explanations.',
    version='1.0.0',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],   # allows Chrome extension requests
    allow_methods=['GET', 'POST'],
    allow_headers=['*'],
)


# ── Load model at startup ─────────────────────────────────────────────────────

MODELS_DIR = Path(__file__).parent.parent / 'models'

def _find_latest_bundle() -> Path:
    bundles = sorted(MODELS_DIR.glob('deploy_*.pkl'))
    if not bundles:
        raise FileNotFoundError(
            f'No deployment bundle found in {MODELS_DIR}. '
            'Run 03_modeling.ipynb to generate one.'
        )
    return bundles[-1]   # latest by filename (timestamp suffix)


bundle_path = os.environ.get('MODEL_PATH') or _find_latest_bundle()
predictor   = Predictor(bundle_path)


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    url: str

    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError('URL must not be empty')
        if not (v.startswith('http://') or v.startswith('https://')):
            v = 'http://' + v   # prepend scheme if missing
        return v


class FeatureContribution(BaseModel):
    feature:    str
    shap_value: float
    value:      float


class PredictResponse(BaseModel):
    url:            str
    classification: str          # 'phishing' | 'legitimate'
    confidence:     float        # 0–100 %
    threshold:      float
    reasons:        list[str]
    top_features:   list[FeatureContribution]
    model_name:     str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get('/health')
def health():
    return {'status': 'ok', 'model': predictor.model_name, 'threshold': predictor.threshold}


@app.post('/predict', response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        result = predictor.predict(request.url)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result
