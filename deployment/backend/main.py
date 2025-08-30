import sys
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
import io
import json
from datetime import datetime
from io import StringIO
import logging
import os
from pathlib import Path
import re
import asyncio
from sklearn.preprocessing import StandardScaler

# =========================
# Logging
# =========================
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(sh)
logger.propagate = False

# =========================
# Paths / Artifacts
# =========================
BASE_DIR = Path(__file__).resolve().parent
APP_DIR = BASE_DIR / "app"
FEATURES_ROOT = APP_DIR / "features"
FEATURE_INDICES_CSV = FEATURES_ROOT / "FeatureIndices.csv"
PERFORMANCE_DIR = APP_DIR / "performance"
MODEL_DIR = PERFORMANCE_DIR / "models"

# =========================
# Constants
# =========================
PTM_TYPES = ['Ace', 'Cro', 'Met', 'Suc', 'Glut']
K = 100

# =========================
# Segmentation Logic (Python equivalent of ReadSeqAndSegment.m)
# =========================
def segment_sequence(sequence: str) -> List[Tuple[int, str]]:
    """
    Segment protein sequence around lysine (K) residues.
    Based on Segmentation.py - uses 21-residue window (10 on each side + lysine)
    """
    window = 49
    frame = int(window / 2)  # 10

    segments = []

    for c, aa in enumerate(sequence):
        if aa == 'K':
            pos = c + 1  # 1-based position

            if c < frame:
                # Left boundary case
                left = sequence[:c]
                right = sequence[c+1:c+frame+1]
                mid_k = sequence[c]
                pad_count = frame - c
                pad = 'X' * pad_count
                segment = pad + left + mid_k + right

            elif c + 1 + frame > len(sequence):
                # Right boundary case
                left = sequence[c-frame:c]
                right = sequence[c+1:]
                mid_k = sequence[c]
                pad_count = c + 1 + frame - len(sequence)
                pad = 'X' * pad_count
                segment = left + mid_k + right + pad

            else:
                # Normal case
                left = sequence[c-frame:c]
                right = sequence[c+1:c+frame+1]
                mid_k = sequence[c]
                segment = left + mid_k + right

            segments.append((pos, segment))

    return segments

# =========================
# Feature Extraction Functions
# =========================

def extract_aaf_features(segments: List[str]) -> np.ndarray:
    """
    Extract AAF (Amino Acid Features) based on AAF.m
    Uses physicochemical properties from aaf_information.txt
    21 residues * 5 properties = 105 features
    """
    # AAF information from the provided file
    aaf_info = {
        'A': [-0.591, -1.302, -0.733, 1.570, -0.146],
        'C': [-1.343, 0.465, -0.862, -1.020, -0.255],
        'D': [1.050, 0.302, -3.656, -0.259, -3.242],
        'E': [1.357, -1.453, 1.477, 0.113, -0.837],
        'F': [-1.006, -0.590, 1.891, -0.397, 0.412],
        'G': [-0.384, 1.652, 1.330, 1.045, 2.064],
        'H': [0.336, -0.417, -1.673, -1.474, -0.078],
        'I': [-1.239, -0.547, 2.131, 0.393, 0.816],
        'K': [1.831, -0.561, 0.533, -0.277, 1.648],
        'L': [-1.019, -0.987, -1.505, 1.266, -0.912],
        'M': [-0.663, -1.524, 2.219, -1.005, 1.212],
        'N': [0.945, 0.828, 1.299, -0.169, 0.933],
        'P': [0.189, 2.081, -1.628, 0.421, -1.392],
        'Q': [0.931, -0.179, -3.005, -0.503, -1.853],
        'R': [1.538, -0.055, 1.502, 0.440, 2.897],
        'S': [-0.228, 1.399, -4.760, 0.670, -2.647],
        'T': [-0.032, 0.326, 2.213, 0.908, 1.313],
        'V': [-1.337, -0.279, -0.544, 1.242, -1.262],
        'W': [-0.595, 0.009, 0.672, -2.128, -0.184],
        'Y': [0.260, 0.830, 3.097, -0.838, 1.512],
        'X': [0.0, 0.0, 0.0, 0.0, 0.0]  # For padding
    }

    features = []
    for segment in segments:
        segment_features = []
        for aa in segment:
            segment_features.extend(aaf_info.get(aa, aaf_info['X']))
        features.append(segment_features)

    return np.array(features)

def extract_binary_features(segments: List[str]) -> np.ndarray:
    """
    Extract Binary Encoding features based on BE.m
    One-hot encoding for each amino acid position
    21 residues * 20 amino acids = 420 features
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWYX'
    n_acids = len(amino_acids)

    features = []
    for segment in segments:
        segment_features = []
        for aa in segment:
            # One-hot encoding
            encoding = [0] * n_acids
            if aa in amino_acids:
                idx = amino_acids.index(aa)
                encoding[idx] = 1
            # X and other non-standard amino acids get all zeros
            segment_features.extend(encoding)
        features.append(segment_features)

    return np.array(features)

def extract_cksaap_features(segments: List[str]) -> np.ndarray:
    """
    Extract C5SAAP (Composition of k-Spaced Amino Acid Pairs) based on CKSAAP.m
    Only k=0,1,2,3,4 (C5SAAP)
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWYX'
    k_max = 4  # k = 0,1,2,3,4

    # Generate all possible k-spaced pairs
    k_pairs = []
    for k in range(k_max + 1):
        for aa1 in amino_acids:
            for aa2 in amino_acids:
                if k == 0:
                    pair = aa1 + aa2
                else:
                    pair = aa1 + '_' * k + aa2
                k_pairs.append(pair)

    features = []
    for segment in segments:
        segment_features = [0] * len(k_pairs)

        # Count occurrences of each k-spaced pair
        for k in range(k_max + 1):
            for i in range(len(segment) - k - 1):
                if k == 0:
                    pair = segment[i] + segment[i + 1]
                else:
                    pair = segment[i] + '_' * k + segment[i + k + 1]

                if pair in k_pairs:
                    pair_idx = k_pairs.index(pair)
                    segment_features[pair_idx] += 1

        features.append(segment_features)

    return np.array(features)

def extract_probability_features(segments: List[str], ptm_type: str) -> np.ndarray:
    """
    Extract Sequence Coupling probability features for the specified PTM type.
    Reads from CSV files in app/performance/dataset/ and infers positive/negative labels.
    """
    try:
        dataset_dir = APP_DIR / "dataset"

        # Look for PTM-specific CSV files
        pos_csv_file = dataset_dir / f"{ptm_type.lower()}_pos.csv"
        neg_csv_file = dataset_dir / f"{ptm_type.lower()}_neg.csv"

        if not pos_csv_file.exists():
            logger.error(f"No dataset file found for {pos_csv_file}")
            return np.zeros((len(segments), 48))

        if not neg_csv_file.exists():
            logger.error(f"No dataset file found for {neg_csv_file}")
            return np.zeros((len(segments), 48))

        # Read sequences from CSV
        df = pd.read_csv(pos_csv_file, header=None)
        all_sequences_pos = df.iloc[:, 0].tolist()

        df = pd.read_csv(neg_csv_file, header=None)
        all_sequences_neg = df.iloc[:, 0].tolist()

        # Split into positive and negative
        ptm_pos_sequences = all_sequences_pos
        ptm_neg_sequences = all_sequences_neg

        # Convert to character arrays
        ptm_pos_list = np.array([list(seq) for seq in ptm_pos_sequences])
        ptm_neg_list = np.array([list(seq) for seq in ptm_neg_sequences])

        logger.info(f"[{ptm_type}] Using {len(ptm_pos_list)} positive, {len(ptm_neg_list)} negative sequences")

        # Rest of the probability computation with vectorized operations
        sample_size = len(segments)
        seq_len = 49
        middle = int((seq_len - 1) / 2)
        coupling = np.zeros((sample_size, seq_len))

        for i in range(sample_size):
            feature_sample = np.array(list(segments[i]))

            for j in range(seq_len):
                if j == middle:
                    continue
                elif j == (middle - 1) or j == (middle + 1):
                    # Non-conditional probability - vectorized
                    feature = feature_sample[j]
                    aa_freq_pos = np.sum(ptm_pos_list[:, j] == feature)
                    prob_pos = aa_freq_pos / len(ptm_pos_sequences)

                    aa_freq_neg = np.sum(ptm_neg_list[:, j] == feature)
                    prob_neg = aa_freq_neg / len(ptm_neg_sequences)
                    prob = prob_pos - prob_neg
                    coupling[i, j] = prob
                else:
                    # Conditional probability - vectorized operations
                    if j < middle:
                        feature1 = feature_sample[j]
                        feature2 = feature_sample[j + 1]

                        # Vectorized counting for positive set
                        pos_feature1_match = ptm_pos_list[:, j] == feature1
                        pos_feature2_match = ptm_pos_list[:, j + 1] == feature2
                        pos_both_match = pos_feature1_match & pos_feature2_match

                        aa_freq_feature1_and_feature2_pos = np.sum(pos_both_match)
                        aa_freq_feature2_pos = np.sum(pos_feature2_match)

                        # Vectorized counting for negative set
                        neg_feature1_match = ptm_neg_list[:, j] == feature1
                        neg_feature2_match = ptm_neg_list[:, j + 1] == feature2
                        neg_both_match = neg_feature1_match & neg_feature2_match

                        aa_freq_feature1_and_feature2_neg = np.sum(neg_both_match)
                        aa_freq_feature2_neg = np.sum(neg_feature2_match)

                    else:
                        feature1 = feature_sample[j]
                        feature2 = feature_sample[j - 1]

                        # Vectorized counting for positive set
                        pos_feature1_match = ptm_pos_list[:, j] == feature1
                        pos_feature2_match = ptm_pos_list[:, j - 1] == feature2
                        pos_both_match = pos_feature1_match & pos_feature2_match

                        aa_freq_feature1_and_feature2_pos = np.sum(pos_both_match)
                        aa_freq_feature2_pos = np.sum(pos_feature2_match)

                        # Vectorized counting for negative set
                        neg_feature1_match = ptm_neg_list[:, j] == feature1
                        neg_feature2_match = ptm_neg_list[:, j - 1] == feature2
                        neg_both_match = neg_feature1_match & neg_feature2_match

                        aa_freq_feature1_and_feature2_neg = np.sum(neg_both_match)
                        aa_freq_feature2_neg = np.sum(neg_feature2_match)

                    # Calculate conditional probabilities
                    prob_feature1_and_feature2_pos = aa_freq_feature1_and_feature2_pos / len(ptm_pos_sequences)
                    prob_feature2_pos = aa_freq_feature2_pos / len(ptm_pos_sequences)

                    con_prob_pos = prob_feature1_and_feature2_pos / prob_feature2_pos if prob_feature2_pos > 0 else 0

                    prob_feature1_and_feature2_neg = aa_freq_feature1_and_feature2_neg / len(ptm_neg_sequences)
                    prob_feature2_neg = aa_freq_feature2_neg / len(ptm_neg_sequences)

                    con_prob_neg = prob_feature1_and_feature2_neg / prob_feature2_neg if prob_feature2_neg > 0 else 0

                    coupling[i, j] = con_prob_pos - con_prob_neg

        sequence_coupling = np.delete(coupling, middle, axis=1)
        return sequence_coupling

    except Exception as e:
        logger.exception(f"Failed to compute probability features for {ptm_type}: {e}")
        return np.zeros((len(segments), 48))

# =========================
# FastAPI Setup
# =========================
app = FastAPI(
    title="PTM Prediction API - Sequence Processing",
    description="Process protein sequences: segment around lysines, extract features, predict PTMs",
    version="5.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "")
appspot_regex = (
    rf"^https://([a-z0-9-]+-dot-)?{re.escape(PROJECT_ID)}\.[a-z]+\.r\.appspot\.com$"
    if PROJECT_ID else None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:8080", "http://127.0.0.1:8080",
    ],
    allow_origin_regex=appspot_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Schemas
# =========================
class SequenceInput(BaseModel):
    sequence: str
    sequence_id: Optional[str] = None

    @field_validator('sequence')
    @classmethod
    def validate_sequence(cls, v):
        if not isinstance(v, str):
            raise ValueError('Sequence must be a string')
        v = v.upper().strip()
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        if not all(aa in valid_aa for aa in v):
            raise ValueError('Invalid amino acid characters in sequence')
        if len(v) > 10000:
            raise ValueError('Sequence too long (max 10000 aa)')
        if len(v) < 5:
            raise ValueError('Sequence too short (min 5 aa)')
        return v

class PTMPrediction(BaseModel):
    predicted: bool
    probability: float

class SegmentPrediction(BaseModel):
    position: int  # Lysine position in original sequence
    residue: str   # Always 'K'
    segment: str   # The 27-residue segment
    predictions: Dict[str, PTMPrediction]

class PredictionResponse(BaseModel):
    sequence_id: str
    sequence: str
    length: int
    lysine_count: int
    predictions: Dict[str, Dict[str, Any]]
    segments: List[SegmentPrediction]
    processing_time: float
    timestamp: str

class BatchResponse(BaseModel):
    results: List[PredictionResponse]
    total_processed: int
    failed_sequences: List[Dict[str, str]]
    processing_time: float

class ExampleSequence(BaseModel):
    id: str
    sequence: str
    description: str

# =========================
# PTM Predictor
# =========================
class PTMPredictor:
    def __init__(self):
        self.ptm_types = PTM_TYPES
        self.ptm_names = {
            'Ace': 'Acetylation', 'Cro': 'Crotonylation', 'Met': 'Methylation',
            'Suc': 'Succinylation', 'Glut': 'Glutarylation'
        }
        self.models = self._load_models()
        self.scalers = self._load_scalers()
        self.indices = self._load_feature_indices()

    def _load_models(self) -> Dict[str, Any]:
        models = {}
        for ptm in self.ptm_types:
            model_path = MODEL_DIR / f"{ptm}_full_model.joblib"
            if not model_path.exists():
                raise RuntimeError(f"Missing trained model: {model_path}")
            try:
                models[ptm] = joblib.load(str(model_path))
                logger.info(f"Loaded model for {ptm}")
            except Exception as e:
                raise RuntimeError(f"Failed to load model for {ptm}: {e}")
        return models

    def _load_scalers(self) -> Dict[str, StandardScaler]:
        scalers = {}
        for ptm in self.ptm_types:
            scaler_path = MODEL_DIR / f"scaler_{ptm}.sav"
            if scaler_path.exists():
                try:
                    scalers[ptm] = joblib.load(str(scaler_path))
                    logger.info(f"Loaded scaler for {ptm}")
                except Exception as e:
                    logger.warning(f"Failed to load scaler for {ptm}: {e}")
            else:
                logger.warning(f"No scaler found for {ptm}")
        return scalers

    def _load_feature_indices(self) -> Dict[str, np.ndarray]:
        if not FEATURE_INDICES_CSV.exists():
            raise RuntimeError(f"Missing feature indices: {FEATURE_INDICES_CSV}")

        try:
            df = pd.read_csv(FEATURE_INDICES_CSV)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            indices = {}
            for ptm in self.ptm_types:
                if ptm not in df.columns:
                    raise RuntimeError(f"Column '{ptm}' missing in {FEATURE_INDICES_CSV}")
                idx = df[ptm].dropna().astype(int).values
                if len(idx) != K:
                    raise RuntimeError(f"{ptm}: expected {K} indices, found {len(idx)}")
                indices[ptm] = idx

            logger.info(f"Loaded feature indices for all PTMs")
            return indices
        except Exception as e:
            raise RuntimeError(f"Failed to load feature indices: {e}")

    async def predict_from_sequence(self, sequence: str, sequence_id: Optional[str]) -> PredictionResponse:
        """
        Process sequence from scratch:
        1. Segment around lysines
        2. Extract all features
        3. Apply feature selection
        4. Scale features
        5. Predict with models
        """
        t0 = datetime.now()
        seq_id = sequence_id or "seq"

        logger.info(f"Processing sequence of length {len(sequence)}")

        # Step 1: Segment sequence around lysines
        segments_with_pos = segment_sequence(sequence)
        if not segments_with_pos:
            raise HTTPException(status_code=400, detail="No lysine residues found in sequence")

        positions = [pos for pos, _ in segments_with_pos]
        segments = [seg for _, seg in segments_with_pos]

        logger.info(f"Found {len(segments)} lysine positions: {positions}")

        # Step 2: Extract all feature types
        logger.info("Extracting features...")
        aaf_features = extract_aaf_features(segments)  # 49 * 5 = 245
        binary_features = extract_binary_features(segments)  # 49 * 21 = 1029
        cksaap_features = extract_cksaap_features(segments)  # 5 * 21² = 2205

        logger.info(f"Feature shapes - AAF: {aaf_features.shape}, Binary: {binary_features.shape}, CKSAAP: {cksaap_features.shape}")

        # Step 3: Combine features and predict for each PTM
        pred_block = {}
        for ptm in self.ptm_types:
            try:
                logger.info(f"Processing {ptm}...")

                # Extract probability features (20 dimensions based on your training data)
                prob_features = extract_probability_features(segments, ptm) # 48
                logger.info(f"{ptm} probability features shape: {prob_features.shape}")

                combined_features = np.concatenate([aaf_features, prob_features, binary_features,
                                      cksaap_features], axis=1)
                logger.info(f"{ptm} combined features shape: {combined_features.shape}")

                # Apply feature selection
                selected_features = combined_features[:, self.indices[ptm]]
                logger.info(f"{ptm} selected features shape: {selected_features.shape}")

                # Apply scaling if available
                if ptm in self.scalers:
                    selected_features = self.scalers[ptm].transform(selected_features)
                    logger.info(f"{ptm} features scaled")

                # Predict
                model = self.models[ptm]
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(selected_features)
                    prob = prob[:, 1] if prob.ndim == 2 and prob.shape[1] > 1 else prob.ravel()
                elif hasattr(model, "decision_function"):
                    scores = model.decision_function(selected_features)
                    prob = 1 / (1 + np.exp(-scores))  # sigmoid
                else:
                    prob = model.predict(selected_features).astype(float)

                pred = (prob > 0.5).astype(int)
                sites = np.where(pred == 1)[0].tolist()

                pred_block[ptm] = {
                    "name": self.ptm_names[ptm],
                    "sites": sites,
                    "probabilities": prob.astype(float).tolist(),
                    "count": int(np.sum(pred)),
                    "max_probability": float(np.max(prob)) if prob.size else 0.0,
                    "avg_probability": float(np.mean(prob)) if prob.size else 0.0,
                }

                logger.info(f"{ptm} prediction complete: {len(sites)} positive sites")

            except Exception as e:
                logger.exception(f"Failed to process {ptm}: {e}")
                raise HTTPException(status_code=500, detail=f"Prediction failed for {ptm}: {str(e)}")

        # Build segment predictions
        segment_preds = []
        for i, (pos, seg) in enumerate(segments_with_pos):
            seg_predictions = {
                ptm: PTMPrediction(
                    predicted=bool(pred_block[ptm]["probabilities"][i] > 0.5),
                    probability=float(pred_block[ptm]["probabilities"][i])
                )
                for ptm in self.ptm_types
            }
            segment_preds.append(SegmentPrediction(
                position=pos,
                residue="K",
                segment=seg,
                predictions=seg_predictions
            ))

        processing_time = (datetime.now() - t0).total_seconds()
        logger.info(f"Sequence processing completed in {processing_time:.2f}s")

        return PredictionResponse(
            sequence_id=str(seq_id),
            sequence=sequence,
            length=len(sequence),
            lysine_count=len(segments),
            predictions=pred_block,
            segments=segment_preds,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )

# Initialize predictor
predictor = PTMPredictor()

# =========================
# Routes
# =========================
@app.get("/")
async def root():
    return {
        "message": "PTM Prediction API - Real-time sequence processing",
        "version": "5.0.0",
        "process": [
            "1. Segment sequence around lysines (27-residue windows)",
            "2. Extract AAF, Binary, CKSAAP, and Probability features",
            "3. Apply saved feature selection (K=100)",
            "4. Scale features using saved scalers",
            "5. Predict with trained models"
        ],
        "ptms": PTM_TYPES,
    }

@app.post("/api/predict/single", response_model=PredictionResponse)
async def predict_single(input_data: SequenceInput):
    try:
        return await predictor.predict_from_sequence(input_data.sequence, input_data.sequence_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Single prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/batch", response_model=BatchResponse)
async def predict_batch(file: UploadFile = File(...)):
    t0 = datetime.now()
    if not file.filename.lower().endswith(('.fasta', '.fa', '.txt')):
        raise HTTPException(status_code=400, detail="Upload FASTA (.fasta/.fa) or .txt")

    try:
        content = (await file.read()).decode('utf-8')
        sequences = []
        failed = []

        # Parse FASTA
        try:
            from Bio import SeqIO
            fasta_io = StringIO(content)
            for record in SeqIO.parse(fasta_io, 'fasta'):
                sequences.append((record.id, str(record.seq).upper()))
        except Exception:
            # Plain text fallback
            lines = [ln.strip() for ln in content.strip().splitlines() if ln.strip() and not ln.strip().startswith('>')]
            for i, line in enumerate(lines):
                sequences.append((f"Sequence_{i+1}", line.upper()))

        if not sequences:
            raise HTTPException(status_code=400, detail="No valid sequences found")

        results = []
        for seq_id, seq in sequences:
            try:
                res = await predictor.predict_from_sequence(seq, seq_id)
                results.append(res)
            except Exception as e:
                failed.append({"sequence_id": seq_id, "error": str(e)})

        return BatchResponse(
            results=results,
            total_processed=len(results),
            failed_sequences=failed,
            processing_time=(datetime.now() - t0).total_seconds()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/examples", response_model=List[ExampleSequence])
async def get_examples():
    return [
        ExampleSequence(
            id="P04637_p53_Human",
            sequence="MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD",
            description="Human p53 tumor suppressor - 14 lysines for PTM prediction"
        ),
        ExampleSequence(
            id="Test_Lysine_Rich",
            sequence="MKGKGKGKGKGKGKGKGKGKGKGKGKGKGKGKGKGKGKGK",
            description="Lysine-rich test sequence - 20 lysines"
        ),
    ]

@app.post("/api/download/results")
async def download_results(request: dict):
    """Download prediction results as CSV using POST to handle large data."""
    try:
        results_data = request.get("results", [])
        csv_buffer = io.StringIO()

        header = ['Sequence_ID', 'Lysine_Position', 'Segment',
                  'Ace_Prob', 'Ace_Pred', 'Cro_Prob', 'Cro_Pred',
                  'Met_Prob', 'Met_Pred', 'Suc_Prob', 'Suc_Pred',
                  'Glut_Prob', 'Glut_Pred']
        csv_buffer.write(','.join(header) + '\n')

        def write_one(r):
            for seg in r.get('segments', []):
                row = [
                    r['sequence_id'],
                    str(seg['position']),
                    seg['segment']
                ]
                for ptm in PTM_TYPES:
                    pdict = seg['predictions'][ptm]
                    row.extend([f"{pdict['probability']:.4f}", str(pdict['predicted'])])
                csv_buffer.write(','.join(row) + '\n')

        if isinstance(results_data, list):
            for r in results_data:
                write_one(r)
        else:
            write_one(results_data)

        csv_buffer.seek(0)
        return StreamingResponse(
            io.BytesIO(csv_buffer.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=ptm_predictions.csv"}
        )
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(PTM_TYPES),
        "feature_selection": f"K={K}",
        "process": "real-time sequence processing"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)