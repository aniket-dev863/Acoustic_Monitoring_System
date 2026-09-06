import React, { useState } from 'react';
import KPICards from '../components/KPICards';
import { Play, UploadCloud, CheckCircle, AlertTriangle, ArrowRight, RefreshCw, Cpu, Layers } from 'lucide-react';
import { createEvent } from '../services/api';

export default function DashboardPage({ stats, events, onRefresh, onSelectEvent }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [samplePath, setSamplePath] = useState('data/sample/fan_anomaly.wav');
  const [gateType, setGateType] = useState('amrag');
  const [resourceLevel, setResourceLevel] = useState(1.0);
  const [machineType, setMachineType] = useState('fan');
  const [isProcessing, setIsProcessing] = useState(false);
  const [lastResult, setLastResult] = useState(null);

  const handleRunEvaluation = async (e) => {
    e.preventDefault();
    setIsProcessing(true);
    setLastResult(null);

    try {
      const formData = new FormData();
      if (selectedFile) {
        formData.append('file', selectedFile);
      } else {
        formData.append('audio_path', samplePath);
      }
      formData.append('gate_type', gateType);
      formData.append('resource_level', resourceLevel.toString());
      formData.append('machine_type', machineType);

      const res = await createEvent(formData);
      setLastResult(res);
      if (onRefresh) onRefresh();
    } catch (err) {
      alert(`Evaluation failed: ${err.response?.data?.error || err.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Overview KPIs */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-white tracking-tight">System Operational Overview</h1>
          <p className="text-xs text-slate-400">
            Real-time acoustic gate condition-monitoring and downstream ML compute saving metrics.
          </p>
        </div>
        <button
          onClick={onRefresh}
          className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-300 transition"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Refresh</span>
        </button>
      </div>

      <KPICards stats={stats} />

      {/* Interactive Live Gate Simulator Card */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-slate-900/60 border border-slate-800 rounded-xl p-5 backdrop-blur-sm">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <div className="w-7 h-7 rounded-lg bg-emerald-500/20 text-emerald-400 flex items-center justify-center">
                <Play className="w-4 h-4" />
              </div>
              <h2 className="text-base font-semibold text-white">Live Acoustic Gate Simulation</h2>
            </div>
            <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700 font-mono">
              Software Edge Node
            </span>
          </div>

          <form onSubmit={handleRunEvaluation} className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* File selection */}
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Synthetic Benchmark File
                </label>
                <select
                  disabled={!!selectedFile}
                  value={samplePath}
                  onChange={(e) => setSamplePath(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-emerald-500 disabled:opacity-50"
                >
                  <option value="data/sample/fan_anomaly.wav">fan_anomaly.wav (Bearing defect surge)</option>
                  <option value="data/sample/fan_normal.wav">fan_normal.wav (Normal steady hum)</option>
                  <option value="data/sample/bearing_anomaly.wav">bearing_anomaly.wav (Raceway impact train)</option>
                  <option value="data/sample/bearing_normal.wav">bearing_normal.wav (Normal meshing)</option>
                  <option value="data/sample/noise_shift_anomaly.wav">noise_shift_anomaly.wav (Noise domain shift)</option>
                  <option value="data/sample/isolated_transient_clicks.wav">isolated_transient_clicks.wav (Transient clicks)</option>
                  <option value="data/sample/silence.wav">silence.wav (Ambient baseline)</option>
                </select>
              </div>

              {/* Upload custom WAV */}
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Or Upload Custom WAV
                </label>
                <input
                  type="file"
                  accept="audio/wav,audio/*"
                  onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                  className="w-full text-xs text-slate-400 file:mr-2 file:py-2 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-medium file:bg-slate-800 file:text-slate-200 hover:file:bg-slate-700 cursor-pointer"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              {/* Gate Type */}
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Gating Algorithm
                </label>
                <select
                  value={gateType}
                  onChange={(e) => setGateType(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-emerald-500"
                >
                  <option value="amrag">Proposed: AMRAG Gate</option>
                  <option value="amplitude">Baseline A: Fixed Amplitude</option>
                  <option value="band">Baseline B: Fixed Band</option>
                  <option value="snr">Baseline C: Adaptive SNR</option>
                </select>
              </div>

              {/* Machine Type */}
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Machine Type
                </label>
                <select
                  value={machineType}
                  onChange={(e) => setMachineType(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-emerald-500"
                >
                  <option value="fan">Fan</option>
                  <option value="bearing">Bearing</option>
                  <option value="gearbox">Gearbox</option>
                  <option value="slide_rail">Slide Rail</option>
                  <option value="valve">Valve</option>
                </select>
              </div>

              {/* Resource Level Slider */}
              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="text-xs font-medium text-slate-300">
                    Edge Resource Level
                  </label>
                  <span className="text-xs font-mono font-bold text-emerald-400">
                    {Math.round(resourceLevel * 100)}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.1"
                  value={resourceLevel}
                  onChange={(e) => setResourceLevel(parseFloat(e.target.value))}
                  className="w-full accent-emerald-500 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-500">
                  <span>Low Battery (0.1)</span>
                  <span>Full Power (1.0)</span>
                </div>
              </div>
            </div>

            <div className="flex justify-end pt-2">
              <button
                type="submit"
                disabled={isProcessing}
                className="flex items-center space-x-2 px-5 py-2.5 rounded-lg text-sm font-semibold bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-900/30 transition disabled:opacity-50"
              >
                {isProcessing ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Processing Stream...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-current" />
                    <span>Run Edge Gate & Anomaly Check</span>
                  </>
                )}
              </button>
            </div>
          </form>

          {/* Last Result Box */}
          {lastResult && (
            <div className="mt-5 p-4 rounded-lg bg-slate-950 border border-slate-800 animate-fadeIn">
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <div className="flex items-center space-x-2">
                  <span className="text-xs font-medium text-slate-400">Gate Output:</span>
                  {lastResult.gate_decision === 'TRIGGER' ? (
                    <span className="px-2.5 py-0.5 rounded text-xs font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30 flex items-center space-x-1">
                      <AlertTriangle className="w-3.5 h-3.5" />
                      <span>TRIGGER (Event Forwarded to ML)</span>
                    </span>
                  ) : (
                    <span className="px-2.5 py-0.5 rounded text-xs font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 flex items-center space-x-1">
                      <CheckCircle className="w-3.5 h-3.5" />
                      <span>IGNORE (Normal Operating Audio - ML Skipped)</span>
                    </span>
                  )}
                </div>
                <span className="text-xs font-mono text-slate-400">
                  Max Score: {lastResult.gate_summary?.max_score} | Thresh: {lastResult.gate_summary?.threshold}
                </span>
              </div>

              <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                <div>
                  <span className="text-slate-400">Trigger Rate:</span>{' '}
                  <span className="text-slate-200 font-mono">
                    {Math.round((lastResult.gate_summary?.trigger_rate || 0) * 100)}%
                  </span>
                </div>
                <div>
                  <span className="text-slate-400">ML Status:</span>{' '}
                  <span className="text-slate-200 font-semibold">
                    {lastResult.ml_executed ? 'EXECUTED' : 'NOT EXECUTED (Compute Saved)'}
                  </span>
                </div>
                {lastResult.prediction && (
                  <>
                    <div>
                      <span className="text-slate-400">ML Diagnosis:</span>{' '}
                      <span
                        className={`font-bold ${
                          lastResult.prediction.result === 'ANOMALY' ? 'text-red-400' : 'text-emerald-400'
                        }`}
                      >
                        {lastResult.prediction.result}
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-400">Reconstruction/Anomaly Score:</span>{' '}
                      <span className="text-slate-200 font-mono">{lastResult.prediction.anomaly_score}</span>
                    </div>
                  </>
                )}
              </div>
            </div>
          )}
        </div>

        {/* System Architecture Reference */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 backdrop-blur-sm flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-semibold text-white mb-2 flex items-center space-x-2">
              <Layers className="w-4 h-4 text-emerald-400" />
              <span>Gating Architecture</span>
            </h3>
            <p className="text-xs text-slate-400 mb-3">
              The proposed gate acts as a lightweight acoustic filter executed at the simulated edge boundary.
            </p>
            <div className="space-y-2 text-xs font-mono bg-slate-950 p-3 rounded-lg border border-slate-800/80">
              <div className="text-slate-400">1. Audio Frame (1024 / 16kHz)</div>
              <div className="text-slate-400">2. Real FFT &amp; Sub-band Power</div>
              <div className="text-slate-400">3. Adaptive Noise Floor (EMA)</div>
              <div className="text-slate-400">4. Multi-Feature Relevance</div>
              <div className="text-slate-400">5. Resource-Aware Hysteresis</div>
              <div className="pt-2 border-t border-slate-800 flex items-center justify-between text-emerald-400 font-bold">
                <span>Gate Decision:</span>
                <span>TRIGGER / IGNORE</span>
              </div>
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800 text-[11px] text-slate-500">
            DCASE 2026 Task 2: Unsupervised anomalous sound detection under noise shifts.
          </div>
        </div>
      </div>

      {/* Recent Events List */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 backdrop-blur-sm">
        <h2 className="text-base font-semibold text-white mb-3">Recent Audio Ingestion &amp; Gating Log</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
              <tr>
                <th className="p-3">Event ID</th>
                <th className="p-3">Machine</th>
                <th className="p-3">Gate Type</th>
                <th className="p-3">Gate Decision</th>
                <th className="p-3">Max Score</th>
                <th className="p-3">ML Execution</th>
                <th className="p-3">Diagnosis</th>
                <th className="p-3">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300">
              {events && events.length > 0 ? (
                events.map((ev) => (
                  <tr key={ev.id} className="hover:bg-slate-800/30 transition">
                    <td className="p-3 font-mono text-slate-400">#{ev.id}</td>
                    <td className="p-3 font-medium capitalize text-white">{ev.machine_type}</td>
                    <td className="p-3 font-mono text-slate-400">{ev.gate_type}</td>
                    <td className="p-3">
                      {ev.triggered ? (
                        <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30">
                          TRIGGERED
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-slate-800 text-slate-400 border border-slate-700">
                          IGNORED
                        </span>
                      )}
                    </td>
                    <td className="p-3 font-mono">{ev.trigger_score?.toFixed(3)}</td>
                    <td className="p-3">
                      {ev.triggered ? (
                        <span className="text-emerald-400 font-medium">Invoked</span>
                      ) : (
                        <span className="text-slate-500 font-medium">Skipped</span>
                      )}
                    </td>
                    <td className="p-3">
                      {ev.ml_result ? (
                        <span
                          className={`font-semibold ${
                            ev.ml_result === 'ANOMALY' ? 'text-red-400' : 'text-emerald-400'
                          }`}
                        >
                          {ev.ml_result}
                        </span>
                      ) : (
                        <span className="text-slate-500">-</span>
                      )}
                    </td>
                    <td className="p-3">
                      <button
                        onClick={() => onSelectEvent(ev.id)}
                        className="text-emerald-400 hover:text-emerald-300 font-medium flex items-center space-x-1"
                      >
                        <span>Inspect</span>
                        <ArrowRight className="w-3 h-3" />
                      </button>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="8" className="p-4 text-center text-slate-500">
                    No acoustic events recorded yet. Run a simulation above to evaluate an audio clip.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
