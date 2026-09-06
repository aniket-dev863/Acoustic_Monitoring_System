/**
 * API Routes Definition.
 */

const express = require('express');
const router = express.Router();

const audioController = require('../controllers/audioController');
const eventController = require('../controllers/eventController');
const statsController = require('../controllers/statsController');
const experimentController = require('../controllers/experimentController');
const { upload } = require('../middleware/upload');

// Health Check
router.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'acoustic-monitoring-backend',
    timestamp: new Date().toISOString(),
  });
});

// Audio Upload & Retrieval
router.post('/audio/upload', upload.single('file'), audioController.uploadAudio);
router.get('/audio/:id', audioController.getAudioFile);

// Events Pipeline
router.post('/events', upload.single('file'), eventController.createEvent);
router.get('/events', eventController.getEvents);
router.get('/events/:id', eventController.getEventById);

// Statistics & Metrics
router.get('/stats', statsController.getStats);

// Experiments & Benchmark Results
router.get('/experiments', experimentController.getExperiments);

module.exports = router;
