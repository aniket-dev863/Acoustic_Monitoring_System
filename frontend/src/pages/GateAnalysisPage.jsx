import React, { useState } from 'react';
import { Radio, Sliders, Play, CheckCircle2, AlertCircle, BarChart2 } from 'lucide-react';
import { createEvent } from '../services/api';

export default function GateAnalysisPage() {
  const [selectedGate, setSelectedGate] = useState('amrag');
  const [sampleAudio, setSampleAudio] = useState('data/sample/fan_anomaly.wav');
  const [resourceLevel, setResourceLevel] = useState(1.0);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const gates = [
    { id: 'amrag', name: 'Proposed: AMRAG Gate', badge: 'Multi-Band Adaptive' },
    { id: 'amplitude', name: 'Baseline A: Fixed Amplitude', badge: 'RMS Threshold' },
    { id: 'band', name: 'Baseline B: Fixed Band', badge: 'Mid-band FFT' },
    { id: 'snr', name: 'Baseline C: Adaptive SNR', badge: 'Full-band SNR' },
  ];

  const handleRunAnalysis = async () => {
    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append('audio_path', sampleAudio);
      formData.append('gate_type', selectedGate);
      formData.append('resource_level', resourceLevel.toString());
      formData.append('machine_type', 'fan');

      const data = await createEvent(formData);
      setAnalysisResult(data);
    } catch (err) {
      alert(`Gate analysis error: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold text-white tracking-tight">Gate Spectral &amp; Temporal Analysis</h1>
        <p className="text-xs text-slate-400">
          Compare acoustic gate triggering behaviors, sub-band energies, adaptive noise floors, and temporal hysteresis.
        </p>
      </div>

      {/* Control Strip */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 backdrop-blur-sm space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Gate Selector */}
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1">Select Acoustic Gate</label>
            <div className="grid grid-cols-1 gap-1.5">
              {gates.map((g) => (
                <button
                  key={g.id}
                  onClick={() => setSelectedGate(g.id)}
                  className={`text-left px-3 py-2 rounded-lg text-xs font-medium border transition flex items-center justify-between ${
                    selectedGate === g.id
                      ? 'bg-emerald-600/20 text-emerald-300 border-emerald-500/40'
                      : 'bg-slate-950 text-slate-400 border-slate-800 hover:bg-slate-800'
                  }`}
                >
                  <span>{g.name}</span>
                  <span className="text-[10px] opacity-70 font-mono">{g.badge}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Test Signal */}
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1">Acoustic Input Signal</label>
            <select
              value={sampleAudio}
              onChange={(e) => setSampleAudio(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-emerald-500 mb-4"
            >
              <option value="data/sample/fan_anomaly.wav">fan_anomaly.wav (Transient surge + squeal)</option>
              <option value="data/sample/fan_normal.wav">fan_normal.wav (Normal steady blade hum)</option>
              <option value="data/sample/bearing_anomaly.wav">bearing_anomaly.wav (Periodic impact clicks)</option>
              <option value="data/sample/bearing_normal.wav">bearing_normal.wav (Normal shaft rotation)</option>
              <option value="data/sample/noise_shift_anomaly.wav">noise_shift_anomaly.wav (Quiet -&gt; Noisy shift)</option>
              <option value="data/sample/isolated_transient_clicks.wav">isolated_transient_clicks.wav (Isolated clicks)</option>
            </select>

            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 text-xs text-slate-400 space-y-1">
              <div className="font-semibold text-slate-300">Signal Specifications:</div>
              <div>Sample Rate: 16,000 Hz | FFT Window: Hann 1024</div>
              <div>Sub-bands: Low (20-800Hz), Mid (800-3kHz), High (3-8kHz)</div>
            </div>
          </div>

          {/* Resource & Execution */}
          <div className="flex flex-col justify-between">
            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-xs font-medium text-slate-300">Resource Level Modulation</label>
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
              <p className="text-[11px] text-slate-400 mt-1">
                Modulates trigger threshold &amp; confirmation persistence based on available edge compute/battery.
              </p>
            </div>

            <button
              onClick={handleRunAnalysis}
              disabled={isLoading}
              className="mt-4 w-full py-2.5 rounded-lg text-sm font-semibold bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-900/30 transition flex items-center justify-center space-x-2"
            >
              <Play className="w-4 h-4 fill-current" />
              <span>{isLoading ? 'Analyzing...' : 'Run Gate Analysis'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Analysis Output Results */}
      {analysisResult ? (
        <div className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
            <div className="bg-slate-900/60 border border-slate-800 p-4 rounded-xl">
              <div className="text-xs text-slate-400 uppercase">Final Decision</div>
              <div className="mt-1 text-lg font-bold">
                {analysisResult.gate_decision === 'TRIGGER' ? (
                  <span className="text-amber-400 flex items-center space-x-1">
                    <AlertCircle className="w-5 h-5" />
                    <span>TRIGGER</span>
                  </span>
                ) : (
                  <span className="text-emerald-400 flex items-center space-x-1">
                    <CheckCircle2 className="w-5 h-5" />
                    <span>IGNORE</span>
                  </span>
                )}
              </div>
              <div className="text-xs text-slate-500 mt-1">
                {analysisResult.ml_executed ? 'ML Model Executed' : 'ML Execution Avoided'}
              </div>
            </div>

            <div className="bg-slate-900/60 border border-slate-800 p-4 rounded-xl">
              <div className="text-xs text-slate-400 uppercase">Peak Gate Score</div>
              <div className="mt-1 text-2xl font-bold text-white font-mono">
                {analysisResult.gate_summary?.max_score}
              </div>
              <div className="text-xs text-slate-500 mt-1">
                Threshold: {analysisResult.gate_summary?.threshold}
              </div>
            </div>

            <div className="bg-slate-900/60 border border-slate-800 p-4 rounded-xl">
              <div className="text-xs text-slate-400 uppercase">Active Trigger Rate</div>
              <div className="mt-1 text-2xl font-bold text-white font-mono">
                {Math.round((analysisResult.gate_summary?.trigger_rate || 0) * 100)}%
              </div>
              <div className="text-xs text-slate-500 mt-1">Percentage of signal active</div>
            </div>

            <div className="bg-slate-900/60 border border-slate-800 p-4 rounded-xl">
              <div className="text-xs text-slate-400 uppercase">Event Intervals</div>
              <div className="mt-1 text-2xl font-bold text-white font-mono">
                {analysisResult.gate_summary?.events?.length || 0}
              </div>
              <div className="text-xs text-slate-500 mt-1">Confirmed temporal intervals</div>
            </div>
          </div>

          {/* Temporal Intervals Table */}
          {analysisResult.gate_summary?.events?.length > 0 && (
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
              <h3 className="text-sm font-semibold text-white mb-2">Confirmed Temporal Trigger Intervals</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left">
                  <thead className="text-slate-400 border-b border-slate-800">
                    <tr>
                      <th className="py-2">Interval</th>
                      <th className="py-2">Start (sec)</th>
                      <th className="py-2">End (sec)</th>
                      <th className="py-2">Duration</th>
                      <th className="py-2">Peak Score</th>
                      <th className="py-2">Mean Score</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/40 text-slate-300 font-mono">
                    {analysisResult.gate_summary.events.map((ev, idx) => (
                      <tr key={idx}>
                        <td className="py-2 font-sans font-medium text-slate-400">Event #{idx + 1}</td>
                        <td className="py-2">{ev.start_time}s</td>
                        <td className="py-2">{ev.end_time}s</td>
                        <td className="py-2">{ev.duration_sec}s</td>
                        <td className="py-2 text-amber-400">{ev.peak_score}</td>
                        <td className="py-2">{ev.mean_score}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="p-8 text-center bg-slate-900/40 border border-slate-800 rounded-xl text-slate-500 text-sm">
          Click "Run Gate Analysis" above to evaluate signal acoustics through the selected gate algorithm.
        </div>
      )}
    </div>
  );
}
