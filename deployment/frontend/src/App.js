// src/App.js
import React, { useState, useEffect } from "react";
import "./App.css";
import api, { API_BASE_URL } from "./api"; // axios instance with baseURL

const PTM_COLORS = {
    Ace: "#ff6b6b",
    Cro: "#4ecdc4",
    Met: "#45b7d1",
    Suc: "#ffa726",
    Glut: "#9c27b0",
};

const PTM_NAMES = {
    Ace: "Acetylation",
    Cro: "Crotonylation",
    Met: "Methylation",
    Suc: "Succinylation",
    Glut: "Glutarylation",
};

// Safe number helpers (no more .toFixed on undefined)
const num = (v, d = 0) => {
    const n = Number(v);
    return Number.isFinite(n) ? n : d;
};
const pct = (v) => `${(num(v) * 100).toFixed(1)}%`;
const sec = (v) => (Number.isFinite(v) ? v.toFixed(2) : "—");

function App() {
    const [sequence, setSequence] = useState("");
    const [sequenceId, setSequenceId] = useState("");
    const [results, setResults] = useState(null);
    const [batchResults, setBatchResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [selectedFile, setSelectedFile] = useState(null);
    const [examples, setExamples] = useState([]);
    const [activeTab, setActiveTab] = useState("single");

    useEffect(() => {
        loadExamples();
    }, []);

    const loadExamples = async () => {
        try {
            const { data } = await api.get("/examples");
            setExamples(data || []);
        } catch (err) {
            console.error("Failed to load examples:", err);
        }
    };

    const handleSinglePrediction = async () => {
        if (!sequence.trim()) {
            setError("Please enter a protein sequence");
            return;
        }
        setLoading(true);
        setError("");
        setResults(null);

        try {
            const { data } = await api.post("/predict/single", {
                sequence: sequence.trim(),
                sequence_id: sequenceId || undefined,
            });
            setResults(data);
        } catch (err) {
            setError(err?.response?.data?.detail || "Prediction failed");
        } finally {
            setLoading(false);
        }
    };

    const handleBatchPrediction = async () => {
        if (!selectedFile) {
            setError("Please select a FASTA file");
            return;
        }
        setLoading(true);
        setError("");
        setBatchResults(null);

        try {
            const formData = new FormData();
            formData.append("file", selectedFile);

            const { data } = await api.post("/predict/batch", formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });
            setBatchResults(data);
        } catch (err) {
            setError(err?.response?.data?.detail || "Batch prediction failed");
        } finally {
            setLoading(false);
        }
    };

    const handleExampleSelect = (example) => {
        setSequence(example.sequence);
        setSequenceId(example.id);
        setActiveTab("single");
    };

    const downloadResults = async (resultsData) => {
        try {
            const { data } = await api.get("/download/results", {
                params: { results: JSON.stringify(resultsData) },
                responseType: "blob",
            });
            const url = window.URL.createObjectURL(new Blob([data]));
            const link = document.createElement("a");
            link.href = url;
            link.setAttribute("download", "ptm_predictions.csv");
            document.body.appendChild(link);
            link.click();
            link.remove();
        } catch (err) {
            console.error("Download failed:", err);
        }
    };

    const renderSequenceVisualization = (result) => {
        const seq = result?.sequence || "";
        const segments = result?.segments || [];

        return (
            <div className="sequence-visualization">
                <h3>Sequence Visualization</h3>
                <div className="sequence-display">
                    {seq.split("").map((aa, index) => {
                        const segment = segments[index];
                        if (!segment || !segment.predictions) {
                            return (
                                <span key={index} className="amino-acid">
                  {aa}
                </span>
                            );
                        }

                        const preds = segment.predictions || {};
                        const maxProb = Math.max(0, ...Object.values(preds).map((p) => num(p?.probability)));
                        const dominantPTM =
                            Object.entries(preds).find(([, p]) => num(p?.probability) === maxProb)?.[0] || "";

                        return (
                            <span
                                key={index}
                                className={`amino-acid ${
                                    maxProb > 0.5 ? `ptm-${String(dominantPTM).toLowerCase()}` : ""
                                }`}
                                title={`Position: ${index + 1}, AA: ${aa}\n${Object.entries(preds)
                                    .map(([ptm, pred]) => `${PTM_NAMES[ptm]}: ${pct(pred?.probability)}`)
                                    .join("\n")}`}
                            >
                {aa}
              </span>
                        );
                    })}
                </div>
                <div className="legend">
                    {Object.entries(PTM_COLORS).map(([ptm, color]) => (
                        <div key={ptm} className="legend-item">
                            <span className="legend-color" style={{ backgroundColor: color }} />
                            <span>{PTM_NAMES[ptm]}</span>
                        </div>
                    ))}
                </div>
            </div>
        );
    };

    const renderPTMSummary = (result) => {
        if (!result?.predictions) return null;

        return (
            <div className="ptm-summary">
                <h3>PTM Prediction Summary</h3>
                <div className="summary-grid">
                    {Object.entries(result.predictions).map(([ptm, data]) => {
                        const sites = Array.isArray(data?.sites) ? data.sites : [];
                        return (
                            <div key={ptm} className="summary-card">
                                <div className="ptm-header">
                                    <h4 style={{ color: PTM_COLORS[ptm] }}>{PTM_NAMES[ptm]}</h4>
                                    <span className="site-count">{num(data?.count, 0)} sites</span>
                                </div>
                                <div className="ptm-stats">
                                    <div className="stat">
                                        <span className="stat-label">Max Probability:</span>
                                        <span className="stat-value">{pct(data?.max_probability)}</span>
                                    </div>
                                    <div className="stat">
                                        <span className="stat-label">Avg Probability:</span>
                                        <span className="stat-value">{pct(data?.avg_probability)}</span>
                                    </div>
                                </div>
                                {sites.length > 0 && (
                                    <div className="predicted-sites">
                                        <span className="stat-label">Predicted Sites:</span>
                                        <div className="sites-list">
                                            {sites.slice(0, 10).map((site) => (
                                                <span key={site} className="site-badge">
                          {site + 1}
                        </span>
                                            ))}
                                            {sites.length > 10 && (
                                                <span className="more-sites">+{sites.length - 10} more</span>
                                            )}
                                        </div>
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </div>
            </div>
        );
    };

    const renderDetailedResults = (result) => {
        const highConfidenceSites =
            (result?.segments ?? []).filter((segment) =>
                Object.values(segment?.predictions ?? {}).some((pred) => num(pred?.probability) > 0.7)
            ) || [];

        return (
            <div className="detailed-results">
                <h3>High Confidence Predictions (&gt;70%)</h3>
                {highConfidenceSites.length > 0 ? (
                    <div className="results-table-container">
                        <table className="results-table">
                            <thead>
                            <tr>
                                <th>Position</th>
                                <th>Residue</th>
                                <th>Acetylation</th>
                                <th>Crotonylation</th>
                                <th>Methylation</th>
                                <th>Succinylation</th>
                                <th>Glutarylation</th>
                            </tr>
                            </thead>
                            <tbody>
                            {highConfidenceSites.map((segment) => (
                                <tr key={segment.position}>
                                    <td>{segment.position}</td>
                                    <td className="residue-cell">{segment.residue}</td>
                                    {Object.entries(segment?.predictions ?? {}).map(([ptm, pred]) => (
                                        <td key={ptm} className="probability-cell">
                                            <div className="probability-bar">
                                                <div
                                                    className="probability-fill"
                                                    style={{
                                                        width: `${num(pred?.probability) * 100}%`,
                                                        backgroundColor: PTM_COLORS[ptm],
                                                    }}
                                                />
                                                <span className="probability-text">{pct(pred?.probability)}</span>
                                            </div>
                                        </td>
                                    ))}
                                </tr>
                            ))}
                            </tbody>
                        </table>
                    </div>
                ) : (
                    <p className="no-results">No high confidence predictions found.</p>
                )}
            </div>
        );
    };

    const renderBatchResults = () => {
        if (!batchResults) return null;

        return (
            <div className="batch-results">
                <div className="batch-summary">
                    <h3>Batch Processing Results</h3>
                    <div className="batch-stats">
                        <div className="stat-item">
                            <span className="stat-number">{num(batchResults?.total_processed, 0)}</span>
                            <span className="stat-label">Sequences Processed</span>
                        </div>
                        <div className="stat-item">
                            <span className="stat-number">{num(batchResults?.failed_sequences?.length, 0)}</span>
                            <span className="stat-label">Failed</span>
                        </div>
                        <div className="stat-item">
                            <span className="stat-number">{sec(batchResults?.processing_time)}s</span>
                            <span className="stat-label">Processing Time</span>
                        </div>
                    </div>

                    <button className="download-btn" onClick={() => downloadResults(batchResults.results)}>
                        Download All Results (CSV)
                    </button>
                </div>

                {batchResults?.failed_sequences?.length > 0 && (
                    <div className="failed-sequences">
                        <h4>Failed Sequences</h4>
                        {batchResults.failed_sequences.map((failed, index) => (
                            <div key={index} className="failed-item">
                                <strong>{failed.sequence_id}:</strong> {failed.error}
                            </div>
                        ))}
                    </div>
                )}

                <div className="batch-results-list">
                    {(batchResults?.results ?? []).map((result, index) => (
                        <div key={index} className="batch-result-item">
                            <div className="result-header">
                                <h4>{result.sequence_id}</h4>
                                <span className="sequence-info">
                  Length: {num(result?.length, 0)} | Processing: {sec(result?.processing_time)}s
                </span>
                            </div>

                            <div className="result-summary">
                                {Object.entries(result?.predictions ?? {}).map(([ptm, data]) => (
                                    <div key={ptm} className="ptm-quick-stat">
                    <span
                        className="ptm-indicator"
                        style={{ backgroundColor: PTM_COLORS[ptm] }}
                    />
                                        <span>
                      {PTM_NAMES[ptm]}: {num(data?.count, 0)} sites
                    </span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        );
    };

    return (
        <div className="app">
            <header className="app-header">
                <h1>PTM Prediction Server</h1>
                <p>Multi-label Post-Translational Modification Prediction</p>
            </header>

            <div className="container">
                {error && (
                    <div className="error-message">
                        <span>⚠️ {error}</span>
                        <button onClick={() => setError("")}>×</button>
                    </div>
                )}

                <div className="tabs">
                    <button
                        className={`tab ${activeTab === "single" ? "active" : ""}`}
                        onClick={() => setActiveTab("single")}
                    >
                        Single Sequence
                    </button>
                    <button
                        className={`tab ${activeTab === "batch" ? "active" : ""}`}
                        onClick={() => setActiveTab("batch")}
                    >
                        Batch Processing
                    </button>
                    <button
                        className={`tab ${activeTab === "examples" ? "active" : ""}`}
                        onClick={() => setActiveTab("examples")}
                    >
                        Examples
                    </button>
                </div>

                {activeTab === "single" && (
                    <div className="input-section">
                        <div className="form-group">
                            <label>Sequence ID (optional):</label>
                            <input
                                type="text"
                                value={sequenceId}
                                onChange={(e) => setSequenceId(e.target.value)}
                                placeholder="Enter sequence identifier"
                                className="text-input"
                            />
                        </div>

                        <div className="form-group">
                            <label>Protein Sequence:</label>
                            <textarea
                                value={sequence}
                                onChange={(e) => setSequence(e.target.value.toUpperCase())}
                                placeholder="Enter protein sequence (single letter amino acid codes)..."
                                className="sequence-textarea"
                                rows={6}
                            />
                            <div className="input-info">
                                Length: {sequence.length} amino acids
                                {sequence.length > 5000 && (
                                    <span className="warning"> (Maximum 5000 allowed)</span>
                                )}
                            </div>
                        </div>

                        <button
                            onClick={handleSinglePrediction}
                            disabled={loading || !sequence.trim()}
                            className="predict-btn"
                        >
                            {loading ? "Predicting..." : "Predict PTM Sites"}
                        </button>
                    </div>
                )}

                {activeTab === "batch" && (
                    <div className="batch-section">
                        <div className="form-group">
                            <label>Upload FASTA File:</label>
                            <input
                                type="file"
                                accept=".fasta,.fa,.txt"
                                onChange={(e) => setSelectedFile(e.target.files[0])}
                                className="file-input"
                            />
                            {selectedFile && (
                                <div className="file-info">
                                    Selected: {selectedFile.name} ({(num(selectedFile?.size / 1024, 0)).toFixed(1)} KB)
                                </div>
                            )}
                        </div>

                        <button
                            onClick={handleBatchPrediction}
                            disabled={loading || !selectedFile}
                            className="predict-btn"
                        >
                            {loading ? "Processing..." : "Process Batch"}
                        </button>

                        <div className="batch-info">
                            <h4>File Format Guidelines:</h4>
                            <ul>
                                <li>FASTA format (.fasta, .fa) or plain text (.txt)</li>
                                <li>Maximum 5000 amino acids per sequence</li>
                                <li>Use standard single-letter amino acid codes</li>
                                <li>Multiple sequences supported in FASTA format</li>
                            </ul>
                        </div>
                    </div>
                )}

                {activeTab === "examples" && (
                    <div className="examples-section">
                        <h3>Example Sequences</h3>
                        <p>Click on any example to load it for prediction:</p>

                        <div className="examples-grid">
                            {examples.map((example) => (
                                <div
                                    key={example.id}
                                    className="example-card"
                                    onClick={() => handleExampleSelect(example)}
                                >
                                    <div className="example-header">
                                        <h4>{example.id}</h4>
                                    </div>
                                    <div className="example-sequence">
                                        {example.sequence.substring(0, 60)}
                                        {example.sequence.length > 60 && "..."}
                                    </div>
                                    <div className="example-description">{example.description}</div>
                                    <div className="example-info">
                                        Length: {example.sequence.length} amino acids
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {loading && (
                    <div className="loading">
                        <div className="spinner" />
                        <p>Processing your request...</p>
                    </div>
                )}

                {results && activeTab === "single" && (
                    <div className="results-section">
                        <div className="results-header">
                            <h2>Prediction Results</h2>
                            <div className="results-meta">
                                <span>Sequence ID: {results.sequence_id}</span>
                                <span>Length: {results.length} amino acids</span>
                                <span>Processing Time: {sec(results?.processing_time)}s</span>
                                <button
                                    className="download-btn-small"
                                    onClick={() => downloadResults(results)}
                                >
                                    Download CSV
                                </button>
                            </div>
                        </div>

                        {renderPTMSummary(results)}
                        {renderSequenceVisualization(results)}
                        {renderDetailedResults(results)}
                    </div>
                )}

                {batchResults && activeTab === "batch" && renderBatchResults()}
            </div>

            <footer className="app-footer">
                <p>
                    Powered by FastAPI and React.js |{" "}
                    <a href={`${API_BASE_URL}/docs`} target="_blank" rel="noopener noreferrer">
                        API Documentation
                    </a>
                </p>
            </footer>
        </div>
    );
}

export default App;