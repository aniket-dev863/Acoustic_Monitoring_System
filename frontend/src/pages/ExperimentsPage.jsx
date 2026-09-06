import React, { useEffect, useState } from 'react';
import { fetchExperiments } from '../services/api';
import { BarChart3, TrendingUp, Layers, CheckCircle2, ShieldAlert } from 'lucide-react';

export default function ExperimentsPage() {
  const [experiments, setExperiments] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadExperiments();
  }, []);

  const loadExperiments = async () => {
    try {
      const data = await fetchExperiments();
      setExperiments(data);
    } catch (err) {
      console.error('Error fetching experiments:', err);
    } finally {
      setLoading(false);
    }
  };

  const comparison = experiments?.gate_comparison;
  const ablations = experiments?.ablation_study;

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-xl font-bold text-white tracking-tight">
          Research Benchmarks &amp; Empirical Evaluation
        </h1>
        <p className="text-xs text-slate-400">
          Empirical comparison of the 4 acoustic gate systems and 6-stage ablation progression across identical machine audio recordings.
        </p>
      </div>

      {/* Part 1: Four-Way Gate Benchmark Comparison */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 backdrop-blur-sm space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-7 h-7 rounded-lg bg-emerald-500/20 text-emerald-400 flex items-center justify-center">
              <BarChart3 className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-white">
                Gate Benchmark Comparison (Identical Corpus)
              </h2>
              <span className="text-xs text-slate-400">
                Evaluation across Fixed Amplitude, Fixed Band, Adaptive SNR, and Proposed AMRAG Gate.
              </span>
            </div>
          </div>
          <span className="text-xs font-mono bg-slate-800 text-slate-300 px-2 py-0.5 rounded">
            DCASE 2026 Compatible
          </span>
        </div>

        {comparison ? (
          <div className="space-y-6">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
                  <tr>
                    <th className="p-3">Gating System</th>
                    <th className="p-3">Anomaly Recall</th>
                    <th className="p-3">Precision</th>
                    <th className="p-3">F1 Score</th>
                    <th className="p-3">False Positive Rate (FPR)</th>
                    <th className="p-3 text-emerald-400">ML Invocations Avoided</th>
                    <th className="p-3">Mean Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-300">
                  {Object.entries(comparison).map(([key, r]) => (
                    <tr
                      key={key}
                      className={`hover:bg-slate-800/30 transition ${
                        key === 'AMRAG-v1' ? 'bg-emerald-950/20 font-semibold' : ''
                      }`}
                    >
                      <td className="p-3">
                        <div className="text-white font-medium">{key}</div>
                        <div className="text-[10px] text-slate-500 font-mono">{r.gate_type}</div>
                      </td>
                      <td className="p-3 font-mono text-emerald-400">
                        {(r.recall * 100).toFixed(1)}%
                      </td>
                      <td className="p-3 font-mono">{(r.precision * 100).toFixed(1)}%</td>
                      <td className="p-3 font-mono font-bold text-white">
                        {r.f1_score.toFixed(3)}
                      </td>
                      <td className="p-3 font-mono text-amber-400">
                        {(r.false_positive_rate * 100).toFixed(1)}%
                      </td>
                      <td className="p-3 font-mono font-bold text-emerald-400">
                        {r.ml_invocations_avoided_pct.toFixed(1)}%
                      </td>
                      <td className="p-3 font-mono text-slate-400">{r.mean_latency_ms} ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Comparison Chart Graphic */}
            <div className="rounded-lg overflow-hidden border border-slate-800 bg-slate-950 p-2">
              <img
                src="/plots/gate_comparison.png"
                alt="Gate Benchmark Comparison Chart"
                className="w-full h-auto object-cover rounded"
                onError={(e) => {
                  e.target.style.display = 'none';
                }}
              />
            </div>
          </div>
        ) : (
          <div className="p-6 text-center text-slate-500 text-xs">
            {loading ? 'Loading comparison metrics...' : 'No benchmark results available. Run compare_gates.py'}
          </div>
        )}
      </div>

      {/* Part 2: Six-Stage Ablation Study */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 backdrop-blur-sm space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-7 h-7 rounded-lg bg-blue-500/20 text-blue-400 flex items-center justify-center">
              <TrendingUp className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-white">
                Six-Stage Component Ablation Study
              </h2>
              <span className="text-xs text-slate-400">
                Measures the exact contribution of each DSP layer from Energy-Only (Stage A) to Full AMRAG (Stage F).
              </span>
            </div>
          </div>
          <span className="text-xs font-mono bg-slate-800 text-slate-300 px-2 py-0.5 rounded">
            Stages A &rarr; F
          </span>
        </div>

        {ablations && ablations.length > 0 ? (
          <div className="space-y-6">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
                  <tr>
                    <th className="p-3">Stage</th>
                    <th className="p-3">Configuration</th>
                    <th className="p-3">Description</th>
                    <th className="p-3">Recall</th>
                    <th className="p-3">FPR</th>
                    <th className="p-3 text-emerald-400">ML Saved %</th>
                    <th className="p-3">F1 Score</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-300">
                  {ablations.map((item, idx) => (
                    <tr
                      key={idx}
                      className={`hover:bg-slate-800/30 transition ${
                        item.stage === 'Stage F' ? 'bg-emerald-950/20 font-semibold' : ''
                      }`}
                    >
                      <td className="p-3 font-mono font-bold text-white">{item.stage}</td>
                      <td className="p-3 font-medium text-slate-200">{item.stage_name}</td>
                      <td className="p-3 text-slate-400">{item.stage_description}</td>
                      <td className="p-3 font-mono text-emerald-400">
                        {(item.recall * 100).toFixed(1)}%
                      </td>
                      <td className="p-3 font-mono text-amber-400">
                        {(item.false_positive_rate * 100).toFixed(1)}%
                      </td>
                      <td className="p-3 font-mono font-bold text-emerald-400">
                        {item.ml_invocations_avoided_pct.toFixed(1)}%
                      </td>
                      <td className="p-3 font-mono text-white font-bold">
                        {item.f1_score.toFixed(3)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Ablation Chart Graphic */}
            <div className="rounded-lg overflow-hidden border border-slate-800 bg-slate-950 p-2">
              <img
                src="/plots/ablation_study.png"
                alt="Ablation Trends Chart"
                className="w-full h-auto object-cover rounded"
                onError={(e) => {
                  e.target.style.display = 'none';
                }}
              />
            </div>
          </div>
        ) : (
          <div className="p-6 text-center text-slate-500 text-xs">
            {loading ? 'Loading ablation study metrics...' : 'No ablation results available. Run ablation_study.py'}
          </div>
        )}
      </div>
    </div>
  );
}
