import React from 'react';
import { Activity, ShieldAlert, Cpu, CheckCircle2, ZapOff, Percent } from 'lucide-react';

export default function KPICards({ stats }) {
  const summary = stats?.summary || {
    total_recordings: 0,
    triggered_events: 0,
    ignored_recordings: 0,
    anomalies_detected: 0,
    normal_events: 0,
    ml_processing_avoided_pct: 0,
    trigger_rate_pct: 0,
  };

  const cards = [
    {
      title: 'Total Audio Clips',
      value: summary.total_recordings,
      icon: Activity,
      color: 'text-blue-400',
      bg: 'bg-blue-500/10 border-blue-500/20',
      desc: 'Ingested audio files',
    },
    {
      title: 'Gate Triggered',
      value: summary.triggered_events,
      icon: ShieldAlert,
      color: 'text-amber-400',
      bg: 'bg-amber-500/10 border-amber-500/20',
      desc: `${summary.trigger_rate_pct}% trigger rate`,
    },
    {
      title: 'Downstream ML Saved',
      value: `${summary.ml_processing_avoided_pct}%`,
      icon: ZapOff,
      color: 'text-emerald-400',
      bg: 'bg-emerald-500/10 border-emerald-500/20',
      desc: `${summary.ignored_recordings} normal clips filtered`,
    },
    {
      title: 'Anomalies Confirmed',
      value: summary.anomalies_detected,
      icon: Cpu,
      color: 'text-red-400',
      bg: 'bg-red-500/10 border-red-500/20',
      desc: 'Identified by ML model',
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((card, idx) => {
        const Icon = card.icon;
        return (
          <div
            key={idx}
            className={`p-4 rounded-xl border backdrop-blur-sm ${card.bg} transition-all hover:scale-[1.01]`}
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">
                {card.title}
              </span>
              <Icon className={`w-5 h-5 ${card.color}`} />
            </div>
            <div className="mt-2 flex items-baseline justify-between">
              <span className="text-2xl font-bold text-white tracking-tight">{card.value}</span>
            </div>
            <p className="mt-1 text-xs text-slate-400">{card.desc}</p>
          </div>
        );
      })}
    </div>
  );
}
