/**
 * Acoustic Condition-Monitoring Server.
 * Node.js + Express + SQLite orchestration engine.
 */

require('dotenv').config({ path: require('path').resolve(__dirname, '../../.env') });
const express = require('express');
const cors = require('cors');
const morgan = require('morgan');
const path = require('path');
const fs = require('fs');

const apiRoutes = require('./routes/api');
const db = require('./database/db');

const app = express();
const PORT = process.env.PORT || 5001;

// CORS configuration
const allowedOrigins = [
  'http://localhost:5173',
  'http://localhost:3000',
  process.env.CLIENT_URL,
].filter(Boolean);

app.use(
  cors({
    origin: (origin, callback) => {
      // Allow requests with no origin (like mobile apps, curl, postman)
      if (!origin || allowedOrigins.includes(origin) || process.env.NODE_ENV !== 'production') {
        return callback(null, true);
      }
      callback(new Error('CORS policy: Not allowed by CORS'));
    },
    credentials: true,
  })
);

app.use(morgan('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve static uploads (audio, spectrogram images)
const uploadsDir = path.resolve(__dirname, '../uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}
app.use('/uploads', express.static(uploadsDir));

// Serve plots from experiments directory
const plotsDir = path.resolve(__dirname, '../../experiments/plots');
if (fs.existsSync(plotsDir)) {
  app.use('/plots', express.static(plotsDir));
}

// Mount API router under /api
app.use('/api', apiRoutes);

// Root health ping
app.get('/', (req, res) => {
  res.json({
    name: 'Acoustic Monitoring System API',
    status: 'online',
    version: '1.0.0',
    documentation: '/api/health',
  });
});

// Centralized Error Handling Middleware
app.use((err, req, res, next) => {
  console.error('[Error]', err.stack || err.message);
  const status = err.status || 500;
  res.status(status).json({
    error: err.message || 'Internal Server Error',
    status,
    timestamp: new Date().toISOString(),
  });
});

// Start Server
if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`[Backend] Acoustic Monitoring API Server running on port ${PORT}`);
    console.log(`[Backend] Health check: http://localhost:${PORT}/api/health`);
  });
}

module.exports = app;
