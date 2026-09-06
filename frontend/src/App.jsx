import React, { useEffect, useState } from 'react';
import Navbar from './components/Navbar';
import DashboardPage from './pages/DashboardPage';
import GateAnalysisPage from './pages/GateAnalysisPage';
import MachineAnalysisPage from './pages/MachineAnalysisPage';
import EventDetailsPage from './pages/EventDetailsPage';
import ExperimentsPage from './pages/ExperimentsPage';
import { fetchStats, fetchEvents } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [stats, setStats] = useState(null);
  const [events, setEvents] = useState([]);
  const [selectedEventId, setSelectedEventId] = useState(null);
  const [loading, setLoading] = useState(true);

  const loadData = async () => {
    try {
      const [statsData, eventsData] = await Promise.all([
        fetchStats(),
        fetchEvents({ limit: 50 }),
      ]);
      setStats(statsData);
      setEvents(eventsData.events || []);
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleSelectEvent = (id) => {
    setSelectedEventId(id);
    setActiveTab('event-details');
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'dashboard' && (
          <DashboardPage
            stats={stats}
            events={events}
            onRefresh={loadData}
            onSelectEvent={handleSelectEvent}
          />
        )}

        {activeTab === 'gate-analysis' && <GateAnalysisPage />}

        {activeTab === 'machine-analysis' && (
          <MachineAnalysisPage
            stats={stats}
            events={events}
            onSelectEvent={handleSelectEvent}
          />
        )}

        {activeTab === 'event-details' && (
          <EventDetailsPage
            eventId={selectedEventId}
            onBack={() => setActiveTab('dashboard')}
          />
        )}

        {activeTab === 'experiments' && <ExperimentsPage />}
      </main>

      <footer className="border-t border-slate-900 bg-slate-950/80 py-4 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row justify-between items-center gap-2">
          <div>
            Acoustic Condition-Monitoring System &bull; AMRAG-v1 Edge Gate Prototype
          </div>
          <div className="text-[11px] font-mono text-slate-600">
            Compliant with DCASE 2026 Task 2 Unsupervised Machine ASD
          </div>
        </div>
      </footer>
    </div>
  );
}
