/**
 * Statistics and Dashboard Metrics Controller.
 */

const db = require('../database/db');

exports.getStats = (req, res, next) => {
  try {
    const totalRow = db.prepare('SELECT COUNT(*) as count FROM events').get();
    const totalRecordings = totalRow ? totalRow.count : 0;

    const triggeredRow = db.prepare('SELECT COUNT(*) as count FROM events WHERE triggered = 1').get();
    const triggeredEvents = triggeredRow ? triggeredRow.count : 0;

    const ignoredRecordings = totalRecordings - triggeredEvents;

    const anomaliesRow = db.prepare("SELECT COUNT(*) as count FROM predictions WHERE result = 'ANOMALY'").get();
    const anomaliesDetected = anomaliesRow ? anomaliesRow.count : 0;

    const normalsRow = db.prepare("SELECT COUNT(*) as count FROM predictions WHERE result = 'NORMAL'").get();
    const normalEvents = normalsRow ? normalsRow.count : 0;

    const triggerRatePct = totalRecordings > 0 ? (triggeredEvents / totalRecordings) * 100.0 : 0.0;
    const mlAvoidedPct = totalRecordings > 0 ? (ignoredRecordings / totalRecordings) * 100.0 : 0.0;

    // Breakdown by machine type
    const machineBreakdown = db.prepare(`
      SELECT 
        machine_type,
        COUNT(*) as total,
        SUM(triggered) as triggered,
        SUM(CASE WHEN processing_status = 'IGNORED' THEN 1 ELSE 0 END) as ignored
      FROM events
      GROUP BY machine_type
    `).all();

    // Breakdown by gate type
    const gateBreakdown = db.prepare(`
      SELECT 
        gate_type,
        COUNT(*) as total,
        SUM(triggered) as triggered,
        AVG(trigger_score) as avg_score
      FROM events
      GROUP BY gate_type
    `).all();

    res.json({
      summary: {
        total_recordings: totalRecordings,
        triggered_events: triggeredEvents,
        ignored_recordings: ignoredRecordings,
        anomalies_detected: anomaliesDetected,
        normal_events: normalEvents,
        ml_processing_avoided_pct: Math.round(mlAvoidedPct * 10) / 10,
        trigger_rate_pct: Math.round(triggerRatePct * 10) / 10,
      },
      machine_breakdown: machineBreakdown,
      gate_breakdown: gateBreakdown,
    });
  } catch (err) {
    next(err);
  }
};
