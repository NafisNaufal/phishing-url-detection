"""
predictor.py — Feature extraction, model inference, and SHAP-based explanation.

Loaded once at API startup; shared across all requests.
"""

from __future__ import annotations

import re
import ipaddress
import math
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import joblib
import numpy as np
import pandas as pd
import shap
import tldextract


# ── Constants ─────────────────────────────────────────────────────────────────

SUSPICIOUS_KEYWORDS = [
    'login', 'verify', 'secure', 'account', 'update', 'confirm',
    'bank', 'password', 'signin', 'paypal', 'webscr', 'token',
    'auth', 'free', 'bonus', 'prize',
]

# ── Feature extraction ────────────────────────────────────────────────────────

def _extract_parts(url: str) -> tuple[str, str, str]:
    ext = tldextract.extract(url)
    sub_parts = [p for p in (ext.subdomain or '').split('.') if p and p.lower() != 'www']
    domain = ext.domain or ''
    tld = ext.suffix.split('.')[-1].lower() if ext.suffix else 'unknown'
    return '.'.join(sub_parts), domain, tld


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _safe_port(parsed) -> int | None:
    try:
        return parsed.port
    except ValueError:
        return None


def _has_ip(url: str) -> int:
    host = (urlparse(url).hostname or '').strip().strip('[]')
    try:
        ipaddress.ip_address(host)
        return 1
    except ValueError:
        return 0


def extract_features(url: str) -> dict:
    """Extract all 25 features from a raw URL string."""
    parsed   = urlparse(url)
    sub, domain, tld = _extract_parts(url)
    path     = parsed.path or ''
    query    = parsed.query or ''

    url_len  = len(url)
    n_letters = len(re.findall(r'[A-Za-z]', url))
    n_digits  = len(re.findall(r'[0-9]', url))

    return {
        # Length
        'URLLength':              url_len,
        'DomainLength':           len(domain),
        'TLDLength':              len(tld),
        'PathLength':             len(path),
        'PathDepth':              len([p for p in path.split('/') if p]),
        # Subdomain
        'NoOfSubDomain':          len([p for p in sub.split('.') if p]) if sub else 0,
        'SubdomainLength':        len(sub),
        # Character composition
        'NoOfLettersInURL':       n_letters,
        'NoOfDigitsInURL':        n_digits,
        'LetterRatioInURL':       n_letters / url_len if url_len else 0.0,
        'DigitRatioInURL':        n_digits  / url_len if url_len else 0.0,
        'NoOfSpecialCharsInURL':  len(re.findall(r'[^A-Za-z0-9]', url)),
        'NoOfDots':               url.count('.'),
        'NoOfDashInURL':          url.count('-'),
        # Binary indicators
        'IsHTTPS':                int(url.lower().startswith('https')),
        'HasIPAddress':           _has_ip(url),
        'HasHyphenInDomain':      int('-' in domain),
        'HasSuspiciousKeyword':   int(any(kw in url.lower() for kw in SUSPICIOUS_KEYWORDS)),
        'DomainHasNumber':        int(bool(re.search(r'\d', domain))),
        'HasAtSymbol':            int('@' in url),
        'HasDoubleSlashInPath':   int('//' in path),
        'URLHasPort':             int(bool(_safe_port(parsed))),
        'HasHexEncoding':         int(bool(re.search(r'%[0-9a-fA-F]{2}', url))),
        # Complexity
        'DomainEntropy':          _shannon_entropy(domain),
        'NoOfQueryParams':        len(parse_qs(query)),
    }


# ── Explanation templates ─────────────────────────────────────────────────────

_TEMPLATES_PHISHING: dict[str, callable] = {
    'HasSuspiciousKeyword': lambda v: "Contains a suspicious keyword (e.g. 'login', 'verify', 'secure', 'bank')" if v else None,
    'HasIPAddress':         lambda v: "Uses a raw IP address instead of a proper domain name" if v else None,
    'URLLength':            lambda v: f"Unusually long URL ({v:.0f} characters) — used to hide the real destination" if v > 75 else None,
    'IsHTTPS':              lambda v: "Does not use HTTPS encryption" if not v else None,
    'HasHyphenInDomain':    lambda v: "Domain contains hyphens — common in lookalike phishing domains" if v else None,
    'DomainHasNumber':      lambda v: "Domain name contains digits, uncommon for legitimate brands" if v else None,
    'DomainEntropy':        lambda v: f"Domain name appears algorithmically generated (entropy {v:.2f})" if v > 3.5 else None,
    'NoOfSubDomain':        lambda v: f"Deep subdomain chain ({int(v)} levels) used to disguise the real domain" if v > 2 else None,
    'HasAtSymbol':          lambda v: "Contains '@' symbol — used to trick browsers into ignoring the real domain" if v else None,
    'HasDoubleSlashInPath': lambda v: "Path contains '//' — a known obfuscation technique" if v else None,
    'URLHasPort':           lambda v: "Specifies a non-standard port number" if v else None,
    'HasHexEncoding':       lambda v: "URL contains hex-encoded characters (%xx) to hide content" if v else None,
    'NoOfQueryParams':      lambda v: f"High number of query parameters ({int(v)})" if v > 5 else None,
    'PathDepth':            lambda v: f"Very deep path structure ({int(v)} levels)" if v > 5 else None,
    'DigitRatioInURL':      lambda v: f"High digit ratio in URL ({v*100:.1f}%) — may be auto-generated" if v > 0.2 else None,
    'NoOfSpecialCharsInURL':lambda v: f"Many special characters ({int(v)}) — typical of obfuscated URLs" if v > 15 else None,
    'NoOfDots':             lambda v: f"Many dots ({int(v)}) — deep subdomain or path nesting" if v > 5 else None,
    'NoOfDashInURL':        lambda v: f"Many dashes ({int(v)}) — common in lookalike domain names" if v > 4 else None,
    'SubdomainLength':      lambda v: f"Long subdomain string ({int(v)} chars) used to impersonate legitimate sites" if v > 20 else None,
    'TLDLength':            lambda v: f"Unusual top-level domain length ({int(v)} characters)" if v > 4 else None,
    'PathLength':           lambda v: f"Very long URL path ({int(v)} characters)" if v > 100 else None,
}

_TEMPLATES_LEGIT: dict[str, callable] = {
    'HasSuspiciousKeyword': lambda v: "No suspicious keywords detected in the URL" if not v else None,
    'HasIPAddress':         lambda v: "Uses a proper registered domain name" if not v else None,
    'IsHTTPS':              lambda v: "Uses HTTPS encryption" if v else None,
    'HasHyphenInDomain':    lambda v: "Clean domain name with no hyphens" if not v else None,
    'DomainEntropy':        lambda v: f"Domain name appears natural (entropy {v:.2f})" if v <= 3.5 else None,
    'NoOfSubDomain':        lambda v: "Simple, clean subdomain structure" if v <= 1 else None,
    'HasAtSymbol':          lambda v: "No '@' symbol in URL" if not v else None,
    'HasHexEncoding':       lambda v: "No hidden hex encoding in URL" if not v else None,
    'URLHasPort':           lambda v: "Uses standard port (no explicit port in URL)" if not v else None,
    'URLLength':            lambda v: f"Normal URL length ({v:.0f} characters)" if v <= 75 else None,
    'DomainHasNumber':      lambda v: "Domain name contains no digits" if not v else None,
}


def _generate_reasons(shap_row: np.ndarray, feat_vals: np.ndarray,
                      features: list[str], is_phishing: bool, top_n: int = 4) -> list[str]:
    templates = _TEMPLATES_PHISHING if is_phishing else _TEMPLATES_LEGIT
    direction = +1 if is_phishing else -1

    scored = [
        (features[j], direction * shap_row[j], feat_vals[j])
        for j in range(len(features))
        if direction * shap_row[j] > 0
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    reasons: list[str] = []
    for feat, _, val in scored:
        fn = templates.get(feat)
        if fn:
            r = fn(val)
            if r:
                reasons.append(r)
        if len(reasons) == top_n:
            break

    return reasons or ['No dominant features identified for this URL']


# ── Predictor ─────────────────────────────────────────────────────────────────

class Predictor:
    """Loads the saved model bundle and serves predictions with SHAP explanations."""

    def __init__(self, bundle_path: str | Path):
        bundle = joblib.load(bundle_path)

        self.model           = bundle['model']
        self.model_name      = bundle.get('model_name', 'unknown')
        self.features        = bundle['features']
        self.threshold       = bundle['threshold']
        self.skewed_features = bundle['skewed_features']
        self._background     = bundle.get('background', None)

        # Initialise SHAP explainer
        # Tree models: TreeExplainer on the model directly
        # LogReg Pipeline: LinearExplainer on the clf step, with scaler applied to background
        if hasattr(self.model, 'named_steps'):
            # Pipeline (LogisticRegression)
            clf_step    = self.model.named_steps['clf']
            scaler_step = self.model.named_steps['scaler']
            bg_log      = self._background if self._background is not None else pd.DataFrame(
                np.zeros((1, len(self.features))), columns=self.features)
            bg_scaled   = pd.DataFrame(scaler_step.transform(bg_log), columns=self.features)
            self._explainer      = shap.LinearExplainer(clf_step, bg_scaled)
            self._is_pipeline    = True
            self._scaler_step    = scaler_step
        elif hasattr(self.model, 'feature_importances_'):
            self._explainer   = shap.TreeExplainer(self.model)
            self._is_pipeline = False
        else:
            self._explainer   = shap.LinearExplainer(self.model, pd.DataFrame(
                np.zeros((1, len(self.features))), columns=self.features))
            self._is_pipeline = False

        print(f'[Predictor] Loaded model: {self.model_name}  |  threshold: {self.threshold:.4f}')

    def predict(self, url: str) -> dict:
        """Run full pipeline: feature extraction → inference → SHAP → explanation."""
        # 1. Extract features
        raw_feats = extract_features(url)
        feat_df   = pd.DataFrame([raw_feats])[self.features]

        # 2. Log1p transform skewed features (applied to all models)
        feat_log = feat_df.copy()
        for col in self.skewed_features:
            if col in feat_log.columns:
                feat_log[col] = np.log1p(feat_log[col].clip(lower=0))

        # 3. Predict — Pipeline (LogReg) scales internally; tree models use feat_log directly
        prob        = float(self.model.predict_proba(feat_log)[0, 1])
        is_phishing = prob >= self.threshold
        label       = 'phishing' if is_phishing else 'legitimate'

        # 4. SHAP values
        # Pipeline: pass scaler-transformed input to LinearExplainer
        # Tree model: pass feat_log directly to TreeExplainer
        if self._is_pipeline:
            feat_for_shap = pd.DataFrame(
                self._scaler_step.transform(feat_log), columns=self.features)
        else:
            feat_for_shap = feat_log

        sv = self._explainer.shap_values(feat_for_shap)
        if isinstance(sv, list):
            sv = sv[1]
        shap_row = np.array(sv[0])

        # 6. Top SHAP features (raw feature values, not scaled)
        raw_vals = feat_df.values[0]
        feat_shap = sorted(
            [{'feature': self.features[j], 'shap_value': round(float(shap_row[j]), 6),
              'value': round(float(raw_vals[j]), 4)}
             for j in range(len(self.features))],
            key=lambda x: abs(x['shap_value']), reverse=True
        )

        # 7. English explanation
        reasons = _generate_reasons(shap_row, raw_vals, self.features, is_phishing)

        return {
            'url':            url,
            'classification': label,
            'confidence':     round(prob * 100, 2),      # percentage
            'threshold':      round(self.threshold, 4),
            'reasons':        reasons,
            'top_features':   feat_shap[:6],              # top 6 for frontend
            'model_name':     self.model_name,
        }
