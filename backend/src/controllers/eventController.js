/**
 * Event Processing Controller.
 * Implements the core pipeline:
 * Upload -> Validate -> Store -> Run Gate ->
 * (If IGNORE -> persist gate result, stop expensive ML)
 * (If TRIGGER -> persist event -> invoke ML -> persist prediction).
 */

const fs = require('fs');
const path = require('path');
const db = require('../database/db');
const gateService = require('../services/gateService');

exports.createEvent = async (req, res, next) => {
  try {
    let audioPath = null;
    let machineType = req.body.machine_type || 'fan';
    const gateType = req.body.gate_type || 'amrag';
    const resourceLevel = parseFloat(req.body.resource_level || '1.0');

    if (req.file) {
      audioPath = req.file.path;
      // Infer machine type from filename if not explicitly provided
      const fname = req.file.originalname.toLowerCase();
      for (const m of ['fan', 'gearbox', 'bearing', 'slide_rail', 'valve']) {
        if (fname.includes(m)) machineType = m;
      }
    } else if (req.body.audio_path) {
      audioPath = req.body.audio_path;
    } else {
      return res.status(400).json({ error: 'Must provide either uploaded audio file or audio_path' });
    }

    if (!fs.existsSync(audioPath)) {
      return res.status(404).json({ error: `Audio file does not exist: ${audioPath}` });
    }

    // Step 1: Run Acoustic Gate
    const execution = await gateService.processAudioFile(audioPath, {
      gateType,
      resourceLevel,
    });

    const gate = execution.gate;
    const triggeredInt = gate.overall_triggered ? 1 : 0;
    const processingStatus = gate.overall_triggered ? 'TRIGGERED' : 'IGNORED';

    // Step 2: Persist event record
    const insertEvent = db.prepare(`
      INSERT INTO events (
        machine_type, audio_path, timestamp, duration, sample_rate,
        gate_type, triggered, trigger_score, gate_metadata,
        processing_status, spectrogram_path
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    const spectrogramPath = execution.ml_prediction?.spectrogram_path || null;

    const eventResult = insertEvent.run(
      machineType,
      audioPath,
      0.0,
      gate.duration_sec,
      16000,
      gate.gate_type,
      triggeredInt,
      gate.max_trigger_score,
      JSON.stringify(gate.metadata || {}),
      processingStatus,
      spectrogramPath
    );

    const eventId = eventResult.lastInsertRowid;

    // Step 3: If triggered, persist ML prediction
    let predictionRecord = null;
    if (execution.ml_executed && execution.ml_prediction) {
      const ml = execution.ml_prediction;
      const insertPred = db.prepare(`
        INSERT INTO predictions (
          event_id, result, anomaly_score, confidence,
          model_version, processing_time
        ) VALUES (?, ?, ?, ?, ?, ?)
      `);

      const predResult = insertPred.run(
        eventId,
        ml.result || 'MODEL_UNAVAILABLE',
        ml.anomaly_score,
        ml.confidence,
        ml.model_version || 'None',
        ml.processing_time_ms || 0.0
      );

      db.prepare('UPDATE events SET processing_status = ? WHERE id = ?').run('PROCESSED', eventId);

      predictionRecord = {
        id: predResult.lastInsertRowid,
        event_id: eventId,
        result: ml.result,
        anomaly_score: ml.anomaly_score,
        confidence: ml.confidence,
        model_version: ml.model_version,
      };
    }

    res.status(201).json({
      success: true,
      event_id: eventId,
      gate_decision: gate.overall_triggered ? 'TRIGGER' : 'IGNORE',
      gate_summary: {
        gate_type: gate.gate_type,
        gate_version: gate.gate_version,
        triggered: gate.overall_triggered,
        trigger_rate: gate.trigger_rate,
        max_score: gate.max_trigger_score,
        threshold: gate.threshold,
        events: gate.events,
      },
      ml_executed: execution.ml_executed,
      prediction: predictionRecord,
    });
  } catch (err) {
    next(err);
  }
};

exports.getEvents = (req, res, next) => {
  try {
    const { machine_type, triggered, limit = 50, offset = 0 } = req.query;

    let query = `
      SELECT 
        e.id, e.machine_type, e.audio_path, e.duration, e.gate_type,
        e.triggered, e.trigger_score, e.processing_status, e.spectrogram_path, e.created_at,
        p.result AS ml_result, p.anomaly_score, p.confidence, p.model_version, p.processing_time AS ml_processing_time
      FROM events e
      LEFT JOIN predictions p ON e.id = p.event_id
      WHERE 1=1
    `;
    const params = [];

    if (machine_type) {
      query += ' AND e.machine_type = ?';
      params.push(machine_type);
    }
    if (triggered !== undefined) {
      query += ' AND e.triggered = ?';
      params.push(triggered === 'true' || triggered === '1' ? 1 : 0);
    }

    query += ' ORDER BY e.id DESC LIMIT ? OFFSET ?';
    params.push(parseInt(limit, 10), parseInt(offset, 10));

    const rows = db.prepare(query).all(...params);
    res.json({ events: rows });
  } catch (err) {
    next(err);
  }
};

exports.getEventById = (req, res, next) => {
  try {
    const { id } = req.params;

    const event = db.prepare(`
      SELECT 
        e.*,
        p.result AS ml_result, p.anomaly_score, p.confidence, p.model_version, p.processing_time AS ml_processing_time
      FROM events e
      LEFT JOIN predictions p ON e.id = p.event_id
      WHERE e.id = ?
    `).get(id);

    if (!event) {
      return res.status(404).json({ error: `Event not found: ${id}` });
    }

    if (event.gate_metadata) {
      try {
        event.gate_metadata = JSON.parse(event.gate_metadata);
      } catch (e) {
        // Keep as string if parsing fails
      }
    }

    res.json({ event });
  } catch (err) {
    next(err);
  }
};
