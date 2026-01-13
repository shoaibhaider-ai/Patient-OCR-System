import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    const uploadedFile = acceptedFiles[0];
    if (uploadedFile) {
      setFile(uploadedFile);
      setPreview(URL.createObjectURL(uploadedFile));
      setResult(null);
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
    },
    multiple: false
  });

  const handleUpload = async () => {
    if (!file) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8001/extract', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to process image. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <h1>🏥 Patient Data Extraction System</h1>
          <p>Upload a patient document to extract information using OCR</p>
        </header>

        <div className="content">
          <div className="upload-section">
            <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
              <input {...getInputProps()} />
              {preview ? (
                <div className="preview-container">
                  <img src={preview} alt="Preview" className="preview-image" />
                  <p className="file-name">{file.name}</p>
                </div>
              ) : (
                <div className="dropzone-content">
                  <div className="upload-icon">📄</div>
                  <p>Drag & drop an image here, or click to select</p>
                  <p className="file-types">Supported: PNG, JPG, JPEG, GIF, BMP, TIFF</p>
                </div>
              )}
            </div>

            <div className="button-group">
              <button
                onClick={handleUpload}
                disabled={!file || loading}
                className="btn btn-primary"
              >
                {loading ? 'Processing...' : 'Extract Data'}
              </button>
              {(file || result) && (
                <button onClick={handleReset} className="btn btn-secondary">
                  Reset
                </button>
              )}
            </div>

            {error && (
              <div className="alert alert-error">
                <strong>Error:</strong> {error}
              </div>
            )}
          </div>

          {result && (
            <div className="results-section">
              <div className="results-header">
                <h2>Extracted Information</h2>
                <div className="confidence-badge">
                  <span>Confidence: {result.confidence}%</span>
                </div>
              </div>

              <div className="results-grid">
                <div className="result-card">
                  <div className="result-icon">👤</div>
                  <div className="result-content">
                    <label>Patient Name</label>
                    <p>{result.data.name}</p>
                  </div>
                </div>

                <div className="result-card">
                  <div className="result-icon">📍</div>
                  <div className="result-content">
                    <label>Address</label>
                    <p>{result.data.address}</p>
                  </div>
                </div>

                <div className="result-card">
                  <div className="result-icon">📞</div>
                  <div className="result-content">
                    <label>Contact Info</label>
                    <p>{result.data.contact}</p>
                  </div>
                </div>

                <div className="result-card">
                  <div className="result-icon">🆔</div>
                  <div className="result-content">
                    <label>Insurance ID</label>
                    <p>{result.data.insurance_id}</p>
                  </div>
                </div>

                <div className="result-card full-width">
                  <div className="result-icon">🏥</div>
                  <div className="result-content">
                    <label>Diagnosis</label>
                    <p>{result.data.diagnosis}</p>
                  </div>
                </div>
              </div>

              {result.raw_text && (
                <details className="raw-text-section">
                  <summary>View Raw OCR Text</summary>
                  <pre className="raw-text">{result.raw_text}</pre>
                </details>
              )}
            </div>
          )}
        </div>

        <footer className="footer">
          <p>CV Lab Project - Patient Data Extraction OCR System</p>
        </footer>
      </div>
    </div>
  );
}

export default App;

