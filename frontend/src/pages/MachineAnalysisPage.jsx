import React, { useState } from 'react';
import { Cpu, ShieldCheck, AlertOctagon, Filter, Clock } from 'lucide-react';

export default function MachineAnalysisPage({ stats, events, onSelectEvent }) {
  const [selectedMachine, setSelectedMachine] = useState('all');

  const machineTypes = ['all', 'fan', 'bearing', 'gearbox', 'slide_rail', 'valve'];

  const filteredEvents = events.filter((ev) => {
    if (selectedMachine === 'all') return true;
    return ev.machine_type === selectedMachine;
  });

  const breakdown = stats?.machine_breakdown || [];

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-white tracking-tight">Machine Condition &amp; Anomaly Analysis</h1>
          <p className="text-xs text-slate-400">
            Per-machine operational integrity, acoustic condition shifts, and ML anomaly classification.
          </p>
        </div>

        {/* Machine Type Filter */}
        <div className="flex items-center space-x-2">
          <Filter className="w-4 h-4 text-slate-400" />
          <div className="flex bg-slate-900 border border-slate-800 rounded-lg p-1 space-x-1">
            {machineTypes.map((m) => (
              <button
                key={m}
                onClick={() => setSelectedMachine(m)}
                className={`px-2.5 py-1 rounded text-xs capitalize transition ${
                  selectedMachine === m
                    ? 'bg-emerald-600 text-white font-medium'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {m.replace('_', ' ')}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Machine Type Aggregations */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        {['fan', 'bearing', 'gearbox', 'slide_rail', 'valve'].map((mtype) => {
          const item = breakdown.find((b) => b.machine_type === mtype) || { total: 0, triggered: 0, ignored: 0 };
          return (
            <div
              key={mtype}
              onClick={() => setSelectedMachine(mtype)}
              className={`p-3.5 rounded-xl border cursor-pointer transition ${
                selectedMachine === mtype
                  ? 'bg-emerald-950/40 border-emerald-500/50'
                  : 'bg-slate-900/50 border-slate-800 hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold capitalize text-slate-200">
                  {mtype.replace('_', ' ')}
                </span>
                <Cpu className="w-3.5 h-3.5 text-emerald-400" />
              </div>
              <div className="mt-2 text-xl font-bold text-white">{item.total} <span className="text-xs font-normal text-slate-500">clips</span></div>
              <div className="mt-1 flex justify-between text-[11px] text-slate-400">
                <span>Triggered: <strong className="text-amber-400">{item.triggered || 0}</strong></span>
                <span>Filtered: <strong className="text-emerald-400">{item.ignored || 0}</strong></span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Event Records for Selected Machine */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 backdrop-blur-sm">
        <h2 className="text-sm font-semibold text-white mb-3">
          Machine Recordings &amp; Diagnostic Records ({filteredEvents.length})
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
              <tr>
                <th className="p-3">ID</th>
                <th className="p-3">Machine</th>
                <th className="p-3">Gate Outcome</th>
                <th className="p-3">Acoustic Score</th>
                <th className="p-3">ML Anomaly Score</th>
                <th className="p-3">Diagnosis</th>
                <th className="p-3">Model Version</th>
                <th className="p-3">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300">
              {filteredEvents.length > 0 ? (
                filteredEvents.map((ev) => (
                  <tr key={ev.id} className="hover:bg-slate-800/30 transition">
                    <td className="p-3 font-mono text-slate-400">#{ev.id}</td>
                    <td className="p-3 font-medium capitalize text-white">{ev.machine_type}</td>
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
                    <td className="p-3 font-mono">
                      {ev.anomaly_score !== null && ev.anomaly_score !== undefined
                        ? ev.anomaly_score.toFixed(4)
                        : '-'}
                    </td>
                    <td className="p-3">
                      {ev.ml_result === 'ANOMALY' ? (
                        <span className="text-red-400 font-bold flex items-center space-x-1">
                          <AlertOctagon className="w-3.5 h-3.5" />
                          <span>ANOMALY</span>
                        </span>
                      ) : ev.ml_result === 'NORMAL' ? (
                        <span className="text-emerald-400 font-bold flex items-center space-x-1">
                          <ShieldCheck className="w-3.5 h-3.5" />
                          <span>NORMAL</span>
                        </span>
                      ) : (
                        <span className="text-slate-500">Skipped (Normal)</span>
                      )}
                    </td>
                    <td className="p-3 font-mono text-slate-400">{ev.model_version || 'N/A'}</td>
                    <td className="p-3">
                      <button
                        onClick={() => onSelectEvent(ev.id)}
                        className="text-emerald-400 hover:text-emerald-300 font-medium"
                      >
                        Inspect
                      </button>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="8" className="p-6 text-center text-slate-500">
                    No recordings found for the selected machine filter.
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
