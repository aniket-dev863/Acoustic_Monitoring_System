/**
 * Audio Controller.
 * Manages audio uploads, file streaming, and metadata.
 */

const fs = require('fs');
const path = require('path');
const db = require('../database/db');

exports.uploadAudio = async (req, res, next) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No audio file provided in request.' });
    }

    const { machine_type = 'unknown' } = req.body;
    const file = req.file;

    res.status(201).json({
      message: 'Audio file uploaded successfully.',
      file: {
        filename: file.filename,
        original_name: file.originalname,
        size_bytes: file.size,
        path: file.path,
        machine_type,
      },
    });
  } catch (err) {
    next(err);
  }
};

exports.getAudioFile = (req, res, next) => {
  try {
    const { id } = req.params;
    const event = db.prepare('SELECT audio_path FROM events WHERE id = ?').get(id);

    if (!event || !event.audio_path) {
      return res.status(404).json({ error: 'Audio file not found for event id' });
    }

    const audioPath = path.resolve(event.audio_path);
    if (!fs.existsSync(audioPath)) {
      return res.status(404).json({ error: 'Physical audio file does not exist on disk' });
    }

    res.sendFile(audioPath);
  } catch (err) {
    next(err);
  }
};
