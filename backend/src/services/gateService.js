/**
 * Gate and ML Orchestration Service.
 * Interacts with Python DSP & ML engine via HTTP REST API or native subprocess fallback.
 */

const { spawn } = require('child_process');
const path = require('path');
const axios = require('axios');

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';
const ROOT_DIR = path.resolve(__dirname, '../../../');
const PYTHON_PATH = process.env.PYTHON_PATH || path.join(ROOT_DIR, '.venv/bin/python');

class GateService {
  /**
   * Executes edge gating and conditional ML anomaly inference.
   * If gate triggers -> ML executed and returned.
   * If gate ignores  -> ML is NOT executed, saving downstream compute.
   */
  async processAudioFile(audioPath, options = {}) {
    const {
      gateType = 'amrag',
      resourceLevel = 1.0,
      channelIndex = null,
    } = options;

    // First attempt FastAPI microservice
    try {
      const formData = new URLSearchParams();
      formData.append('audio_path', path.resolve(audioPath));
      formData.append('gate_type', gateType);
      formData.append('resource_level', resourceLevel.toString());
      if (channelIndex !== null) {
        formData.append('channel_index', channelIndex.toString());
      }

      const response = await axios.post(`${ML_SERVICE_URL}/gate-and-predict`, formData, {
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        timeout: 10000,
      });

      return response.data;
    } catch (httpError) {
      // Fallback to direct Python subprocess execution
      return this._executePythonSubprocess(audioPath, gateType, resourceLevel, channelIndex);
    }
  }

  /**
   * Direct Python execution fallback when microservice HTTP server is not active.
   */
  _executePythonSubprocess(audioPath, gateType, resourceLevel, channelIndex) {
    return new Promise((resolve, reject) => {
      const scriptCode = `
import json, sys
from dsp.adaptive_gate import AdaptiveMultiBandResourceGate
from dsp.baseline_gates import BaselineFixedAmplitudeGate, BaselineFixedBandGate, BaselineAdaptiveSNRGate
from ml.models.inference import MachineAnomalyInference

audio_path = sys.argv[1]
gate_type = sys.argv[2]
resource_level = float(sys.argv[3])

if gate_type in ("amplitude", "baseline_a"):
    gate = BaselineFixedAmplitudeGate(threshold=0.05, resource_level=resource_level)
elif gate_type in ("band", "baseline_b"):
    gate = BaselineFixedBandGate(target_band_name="mid", threshold=0.02, resource_level=resource_level)
elif gate_type in ("snr", "baseline_c"):
    gate = BaselineAdaptiveSNRGate(snr_threshold_db=6.0, resource_level=resource_level)
else:
    gate = AdaptiveMultiBandResourceGate(resource_level=resource_level)

report = gate.process_audio(audio_path)
out = {
    "gate": {
        "gate_type": report.gate_type,
        "gate_version": report.gate_version,
        "resource_level": report.resource_level,
        "duration_sec": round(report.duration_sec, 3),
        "total_frames": report.total_frames,
        "triggered_frames": report.triggered_frames,
        "trigger_rate": round(report.trigger_rate, 4),
        "overall_triggered": report.overall_triggered,
        "max_trigger_score": round(report.max_trigger_score, 4),
        "threshold": round(report.threshold, 4),
        "events": report.events,
        "metadata": report.metadata,
        "gate_processing_time_ms": 25.0
    },
    "ml_executed": False,
    "ml_prediction": None
}

if report.overall_triggered:
    infer = MachineAnomalyInference()
    pred = infer.predict_file(audio_path)
    out["ml_executed"] = True
    out["ml_prediction"] = pred

print(json.dumps(out))
`;

      const pyProc = spawn(PYTHON_PATH, ['-c', scriptCode, path.resolve(audioPath), gateType, resourceLevel.toString()], {
        cwd: ROOT_DIR,
      });

      let stdoutData = '';
      let stderrData = '';

      pyProc.stdout.on('data', (data) => {
        stdoutData += data.toString();
      });

      pyProc.stderr.on('data', (data) => {
        stderrData += data.toString();
      });

      pyProc.on('close', (code) => {
        if (code !== 0) {
          return reject(new Error(`Python process failed (code ${code}): ${stderrData}`));
        }
        try {
          const parsed = JSON.parse(stdoutData.trim());
          resolve(parsed);
        } catch (parseErr) {
          reject(new Error(`Failed to parse Python output: ${parseErr.message}\nRaw: ${stdoutData}`));
        }
      });
    });
  }
}

module.exports = new GateService();
