"""ESG report classifier using a pre-trained Random Forest model.

Migrated from ESGioannis/pdf_verifier.py.

CRITICAL BUG FIXED: The original code called train_and_save_model() at module
import time (line 143), causing training to run on every import. This version
separates loading from training.

Usage:
    from services.pdf.verifier import verifier
    result = verifier.classify(pdf_path)
    # {"is_esg_report": True, "confidence": 0.92, "features": {...}}

To train a new model:
    from services.pdf.verifier import train_model
    train_model(dataset_dir="./training_data", ...)
"""
from __future__ import annotations

import logging
import os
import threading

import numpy as np
import pdfplumber
from pdfminer.high_level import extract_text
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFParser
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from esg_mcp_servers.settings import settings

logger = logging.getLogger(__name__)


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(file_path: str) -> dict | None:
    """Extract structural and textual features from a PDF for classification."""
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            num_pages = len(pdf.pages)
            font_names: set[str] = set()
            num_headings = 0
            num_bullets = 0
            num_tables = 0
            num_figures = 0

            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text
                for char in page.chars:
                    font_names.add(char.get("fontname", ""))
                for line in page_text.split("\n"):
                    if line.isupper() and len(line.split()) < 10:
                        num_headings += 1
                    if line.startswith(("- ", "* ", "• ")):
                        num_bullets += 1
                num_tables += len(page.find_tables())
                num_figures += len(page.images)

        num_fonts = len(font_names)

        # PDF metadata via pdfminer
        author = title = creation_date = ""
        try:
            with open(file_path, "rb") as f:
                parser = PDFParser(f)
                doc = PDFDocument(parser)
                meta = doc.info[0] if doc.info else {}
                author = str(meta.get("Author", ""))
                title = str(meta.get("Title", ""))
                creation_date = str(meta.get("CreationDate", ""))
        except Exception:
            pass

        return {
            "text": text,
            "num_pages": num_pages,
            "num_fonts": num_fonts,
            "num_headings": num_headings,
            "num_bullets": num_bullets,
            "num_tables": num_tables,
            "num_figures": num_figures,
            "author": author,
            "title": title,
            "creation_date": creation_date,
        }
    except Exception as e:
        logger.error(f"Feature extraction failed for {file_path}: {e}")
        return None


def _vectorize_single(features: dict, vectorizer: TfidfVectorizer) -> np.ndarray:
    text_vec = vectorizer.transform([features["text"]])
    numeric = np.array([[
        features["num_pages"], features["num_fonts"], features["num_headings"],
        features["num_bullets"], features["num_tables"], features["num_figures"],
    ]])
    return np.hstack([text_vec.toarray(), numeric])


def _vectorize_dataset(
    data: list[dict],
    vectorizer: TfidfVectorizer | None = None,
) -> tuple[np.ndarray, TfidfVectorizer]:
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=1000)
        text_vecs = vectorizer.fit_transform([d["text"] for d in data])
    else:
        text_vecs = vectorizer.transform([d["text"] for d in data])
    numeric = np.array([[
        d["num_pages"], d["num_fonts"], d["num_headings"],
        d["num_bullets"], d["num_tables"], d["num_figures"],
    ] for d in data])
    return np.hstack([text_vecs.toarray(), numeric]), vectorizer


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    dataset_dir: str,
    model_path: str | None = None,
    vectorizer_path: str | None = None,
) -> tuple:
    """Train the RandomForest classifier on a labelled dataset.

    dataset_dir must contain subdirectories named 'report' and 'non report'.
    """
    import joblib

    model_path = model_path or settings.PDF_VERIFIER_MODEL_PATH
    vectorizer_path = vectorizer_path or settings.PDF_VERIFIER_VECTORIZER_PATH

    data, labels = [], []
    for label in ["report", "non report"]:
        dir_path = os.path.join(dataset_dir, label)
        if not os.path.isdir(dir_path):
            logger.warning(f"Training dir not found: {dir_path}")
            continue
        for fname in os.listdir(dir_path):
            fpath = os.path.join(dir_path, fname)
            feats = extract_features(fpath)
            if feats:
                data.append(feats)
                labels.append(label)

    if not data:
        raise ValueError("No training data found — check dataset_dir structure.")

    features, vectorizer = _vectorize_dataset(data)
    X_tr, X_te, y_tr, y_te = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)

    logger.info(f"Train accuracy: {model.score(X_tr, y_tr):.3f}")
    logger.info(f"Test  accuracy: {model.score(X_te, y_te):.3f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    logger.info(f"Model saved → {model_path}")
    return model, vectorizer


# ── Inference singleton ───────────────────────────────────────────────────────

class ESGReportVerifier:
    """Thread-safe lazy-loading inference wrapper."""

    _instance: "ESGReportVerifier | None" = None
    _lock = threading.Lock()

    def __init__(self):
        self._model: RandomForestClassifier | None = None
        self._vectorizer: TfidfVectorizer | None = None
        self._init_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "ESGReportVerifier":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load(self):
        if self._model is not None:
            return
        import joblib
        with self._init_lock:
            if self._model is not None:
                return
            mp = settings.PDF_VERIFIER_MODEL_PATH
            vp = settings.PDF_VERIFIER_VECTORIZER_PATH
            if os.path.exists(mp) and os.path.exists(vp):
                self._model = joblib.load(mp)
                self._vectorizer = joblib.load(vp)
                logger.info("ESG report verifier model loaded")
            else:
                logger.warning(
                    f"Verifier model not found at {mp}. "
                    "Call services.pdf.verifier.train_model() first."
                )

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def classify(self, file_path: str) -> dict:
        """Classify a PDF file.

        Returns:
            {"is_esg_report": bool, "confidence": float, "features": dict}
        """
        self._load()

        features = extract_features(file_path)
        if features is None:
            return {"is_esg_report": False, "confidence": 0.0, "features": {}}

        if self._model is None or self._vectorizer is None:
            # Model not available — default to True (optimistic)
            logger.warning("Verifier model unavailable; defaulting is_esg_report=True")
            return {"is_esg_report": True, "confidence": 0.5, "features": features}

        X = _vectorize_single(features, self._vectorizer)
        prediction = self._model.predict(X)[0]
        proba = self._model.predict_proba(X)[0]
        classes = list(self._model.classes_)
        confidence = float(proba[classes.index(prediction)])

        return {
            "is_esg_report": prediction == "report",
            "confidence": confidence,
            "features": {k: v for k, v in features.items() if k != "text"},
        }


# Module-level singleton
verifier = ESGReportVerifier.get_instance()
