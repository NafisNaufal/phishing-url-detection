/* popup.js — Chrome extension frontend logic */

const API_URL = 'http://localhost:8000/predict';

// ── DOM refs ─────────────────────────────────────────────────────────────────
const urlInput      = document.getElementById('urlInput');
const checkBtn      = document.getElementById('checkBtn');
const resultCard    = document.getElementById('resultCard');
const badge         = document.getElementById('badge');
const confidenceText= document.getElementById('confidenceText');
const confidenceBar = document.getElementById('confidenceBar');
const reasonsList   = document.getElementById('reasonsList');
const featuresTable = document.getElementById('featuresTable');
const loading       = document.getElementById('loading');
const errorBox      = document.getElementById('errorBox');

// ── Helpers ──────────────────────────────────────────────────────────────────
function show(el) { el.classList.remove('hidden'); }
function hide(el) { el.classList.add('hidden'); }

function reset() {
  hide(resultCard);
  hide(loading);
  hide(errorBox);
}

function showError(msg) {
  hide(loading);
  hide(resultCard);
  errorBox.textContent = '⚠️ ' + msg;
  show(errorBox);
}

function renderResult(data) {
  hide(loading);
  hide(errorBox);

  const isPhishing = data.classification === 'phishing';

  // Badge
  badge.className  = `badge ${data.classification}`;
  badge.textContent = isPhishing ? '⚠️  Phishing URL Detected' : '✅  Legitimate URL';

  // Confidence bar
  confidenceText.textContent = `${data.confidence.toFixed(1)}%`;
  confidenceBar.style.width  = `${data.confidence}%`;
  confidenceBar.className    = `confidence-bar ${data.classification}`;

  // Reasons
  reasonsList.innerHTML = '';
  data.reasons.forEach(reason => {
    const li = document.createElement('li');
    li.textContent = reason;
    reasonsList.appendChild(li);
  });

  // Top features (SHAP bar chart)
  featuresTable.innerHTML = '';
  const maxShap = Math.max(...data.top_features.map(f => Math.abs(f.shap_value)));

  data.top_features.forEach(f => {
    const pct   = maxShap > 0 ? (Math.abs(f.shap_value) / maxShap) * 100 : 0;
    const dir   = f.shap_value >= 0 ? 'pos' : 'neg';
    const label = f.shap_value >= 0 ? '↑ phishing' : '↓ legit';

    const row = document.createElement('div');
    row.className = 'feat-row';
    row.innerHTML = `
      <span class="feat-name" title="${f.feature}">${f.feature}</span>
      <div class="feat-bar-wrap">
        <div class="feat-bar ${dir}" style="width:${pct.toFixed(1)}%" title="${label}"></div>
      </div>
      <span class="feat-val">${f.value}</span>
    `;
    featuresTable.appendChild(row);
  });

  show(resultCard);
}

// ── Main action ──────────────────────────────────────────────────────────────
async function checkUrl() {
  const url = urlInput.value.trim();
  if (!url) { showError('Please enter a URL.'); return; }

  reset();
  show(loading);
  checkBtn.disabled = true;

  try {
    const res = await fetch(API_URL, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ url }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error (${res.status})`);
    }

    const data = await res.json();
    renderResult(data);

  } catch (err) {
    if (err.name === 'TypeError') {
      showError('Cannot reach the API. Make sure the server is running on localhost:8000.');
    } else {
      showError(err.message);
    }
  } finally {
    checkBtn.disabled = false;
  }
}

// ── Event listeners ──────────────────────────────────────────────────────────
checkBtn.addEventListener('click', checkUrl);
urlInput.addEventListener('keydown', e => { if (e.key === 'Enter') checkUrl(); });
