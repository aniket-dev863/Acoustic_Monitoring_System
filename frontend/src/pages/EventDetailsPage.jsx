import React, { useEffect, useState } from 'react';
import { fetchEventById } from '../services/api';
import { Play, Volume2, Database, ShieldAlert, CheckCircle2, Clock, Cpu, FileAudio } from 'lucide-react';

export default function EventDetailsPage({ eventId, onBack }) {
  const [eventData, setEventData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (eventId) {
      loadEvent(eventId);
    }
  }, [eventId]);

  const loadEvent = async (id) => {
    setLoading(true);
    try {
      const res = await fetchEventById(id);
      setEventData(res.event);
    } catch (err) {
      console.error('Error fetching event details:', err);
    } finally {
      setLoading(false);
    }
  };

  if (!eventId) {
    return (
      <div className="p-12 text-center bg-slate-900/40 border border-slate-800 rounded-xl text-slate-500">
        <Database className="w-12 h-12 mx-auto mb-3 opacity-40 text-slate-400" />
        <p className="text-sm">Select an event from the Dashboard or Machine Analysis tabs to inspect full details.</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="p-12 text-center text-slate-400">
        Loading event #{eventId} data...
      </div>
    );
  }

  if (!eventData) {
    return (
      <div className="p-8 text-center text-red-400">
        Event #{eventId} could not be loaded.
      </div>
    );
  }

  const meta = eventData.gate_metadata || {};
  const audioUrl = `/api/audio/${eventData.id}`;
  const specUrl = eventData.spectrogram_path ? `/${eventData.spectrogram_path}` : null;

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <div className="flex items-center space-x-2">
            <span className="text-xs font-mono text-emerald-400 bg-emerald-950 px-2 py-0.5 rounded border border-emerald-800">
              Event #{eventData.id}
            </span>
            <span className="text-xs text-slate-400 font-medium capitalize">
              Machine: <strong className="text-white">{eventData.machine_type}</strong>
            </span>
          </div>
          <h1 className="text-xl font-bold text-white tracking-tight mt-1">
            Acoustic Event Diagnostic Inspector
          </h1>
        </div>

        {onBack && (
          <button
            onClick={onBack}
            className="px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 transition"
          >
            &larr; Back to Events
          </button>
        )}
      </div>

      {/* Primary Status Strip */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {/* Gate Outcome */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
          <span className="text-xs text-slate-400 uppercase tracking-wider">Gate Outcome</span>
          <div className="mt-1 flex items-center space-x-2">
            {eventData.triggered ? (
              <span className="text-amber-400 font-bold text-lg flex items-center space-x-1.5">
                <ShieldAlert className="w-5 h-5" />
                <span>TRIGGERED</span>
              </span>
            ) : (
              <span className="text-emerald-400 font-bold text-lg flex items-center space-x-1.5">
                <CheckCircle2 className="w-5 h-5" />
                <span>IGNORED</span>
              </span>
            )}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">
            {eventData.triggered ? 'Forwarded to deep ML anomaly model' : 'Normal operating sound; ML bypassed'}
          </p>
        </div>

        {/* ML Diagnosis */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
          <span className="text-xs text-slate-400 uppercase tracking-wider">ML Anomaly Result</span>
          <div className="mt-1 text-lg font-bold">
            {eventData.ml_result === 'ANOMALY' ? (
              <span className="text-red-400">ANOMALY DETECTED</span>
            ) : eventData.ml_result === 'NORMAL' ? (
              <span className="text-emerald-400">NORMAL MACHINE STATE</span>
            ) : (
              <span className="text-slate-500">Not Invoked (Compute Saved)</span>
            )}
          </div>
          <p className="text-[11px] text-slate-500 mt-1 font-mono">
            Score: {eventData.anomaly_score !== null ? eventData.anomaly_score : 'N/A'} | Model: {eventData.model_version || 'None'}
          </p>
        </div>

        {/* Audio Duration */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
          <span className="text-xs text-slate-400 uppercase tracking-wider">Clip Duration &amp; Sample Rate</span>
          <div className="mt-1 text-lg font-bold text-white font-mono">
            {eventData.duration?.toFixed(2)}s @ {eventData.sample_rate} Hz
          </div>
          <p className="text-[11px] text-slate-500 mt-1 font-mono">
            Recorded: {eventData.created_at}
          </p>
        </div>
      </div>

      {/* Audio Playback & Spectrogram Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Audio Player Card */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 backdrop-blur-sm space-y-4">
          <div className="flex items-center space-x-2">
            <Volume2 className="w-4 h-4 text-emerald-400" />
            <h2 className="text-sm font-semibold text-white">Audio Playback</h2>
          </div>

          <div className="p-4 bg-slate-950 rounded-lg border border-slate-800">
            <audio controls className="w-full" src={audioUrl}>
              Your browser does not support the audio element.
            </audio>
            <div className="mt-2 text-[11px] text-slate-500 truncate font-mono">
              Source: {eventData.audio_path}
            </div>
          </div>

          {/* Explainable Gate Metadata */}
          <div>
            <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
              Explainable DSP Gate Metadata
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
              <div className="bg-slate-950 p-2.5 rounded border border-slate-800">
                <span className="text-slate-500 text-[10px] block">GATE SCORE</span>
                <span className="font-mono text-white font-bold">{eventData.trigger_score?.toFixed(4)}</span>
              </div>
              <div className="bg-slate-950 p-2.5 rounded border border-slate-800">
                <span className="text-slate-500 text-[10px] block">THRESHOLD</span>
                <span className="font-mono text-slate-300">{meta.threshold || '0.55'}</span>
              </div>
              <div className="bg-slate-950 p-2.5 rounded border border-slate-800">
                <span className="text-slate-500 text-[10px] block">SNR (dB)</span>
                <span className="font-mono text-slate-300">{meta.snr_db ?? 'N/A'}</span>
              </div>
              <div className="bg-slate-950 p-2.5 rounded border border-slate-800">
                <span className="text-slate-500 text-[10px] block">NOISE FLOOR</span>
                <span className="font-mono text-slate-300">{meta.noise_floor ?? 'N/A'}</span>
              </div>
              <div className="bg-slate-950 p-2.5 rounded border border-slate-800">
                <span className="text-slate-500 text-[10px] block">DOMINANT FREQ</span>
                <span className="font-mono text-slate-300">{meta.dominant_frequency ? `${meta.dominant_frequency} Hz` : 'N/A'}</span>
              </div>
              <div className="bg-slate-950 p-2.5 rounded border border-slate-800">
                <span className="text-slate-500 text-[10px] block">BAND ENERGY RATIO</span>
                <span className="font-mono text-slate-300">{meta.band_energy_ratio ?? 'N/A'}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Mel Spectrogram Preview */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 backdrop-blur-sm space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Cpu className="w-4 h-4 text-emerald-400" />
              <h2 className="text-sm font-semibold text-white">Log-Mel Spectrogram (ML Input)</h2>
            </div>
            <span className="text-[10px] font-mono text-slate-400 bg-slate-800 px-2 py-0.5 rounded">
              128 Mels / 1024 FFT
            </span>
          </div>

          {specUrl ? (
            <div className="rounded-lg overflow-hidden border border-slate-800 bg-slate-950">
              <img
                src={specUrl}
                alt="Log-Mel Spectrogram"
                className="w-full h-auto object-cover"
                onError={(e) => {
                  e.target.style.display = 'none';
                }}
              />
            </div>
          ) : (
            <div className="h-48 flex items-center justify-center rounded-lg border border-dashed border-slate-800 text-slate-500 text-xs p-4 text-center">
              Spectrogram generated only when acoustic gate triggers downstream ML inference.
            </div>
          )}

          <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 text-xs text-slate-400 space-y-1">
            <div className="font-semibold text-slate-300">Processing Latency Breakdown:</div>
            <div className="flex justify-between">
              <span>Edge Gate Latency:</span>
              <span className="font-mono text-slate-200">~25.0 ms</span>
            </div>
            {eventData.ml_processing_time && (
              <div className="flex justify-between">
                <span>ML Anomaly Latency:</span>
                <span className="font-mono text-slate-200">{eventData.ml_processing_time} ms</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
