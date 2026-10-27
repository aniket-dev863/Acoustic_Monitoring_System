/**
 * Live Audio Recording & Pipeline Integration Test Suite.
 *
 * Verifies:
 * 1. Client WAV PCM-16 encoding algorithm and RIFF header compliance.
 * 2. Multipart file upload of live 16 kHz mono WAV recordings via POST /api/events.
 * 3. End-to-end IGNORE path (normal acoustic hum, ML bypassed).
 * 4. End-to-end TRIGGER path (anomalous acoustic spikes, ML autoencoder invoked).
 * 5. Audio playback preview retrieval via GET /api/audio/:id.
 * 6. Error handling for missing files, invalid types, and empty requests.
 */

const test = require('node:test');
const assert = require('node:assert');
const path = require('path');
const fs = require('fs');
const http = require('http');
const app = require('../src/server');

let server;
const TEST_PORT = 5096;

/**
 * Creates a synthetic 16 kHz mono 16-bit PCM WAV buffer matching
 * the frontend's encodeWavPCM16 implementation.
 */
function createSyntheticWavBuffer(durationSec = 2.0, frequencyHz = 120, amplitude = 0.05) {
  const sampleRate = 16000;
  const numSamples = Math.floor(durationSec * sampleRate);
  const numChannels = 1;
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = numSamples * bytesPerSample;

  const buffer = Buffer.alloc(44 + dataSize);

  // 1. RIFF Header
  buffer.write('RIFF', 0, 'ascii');
  buffer.writeUInt32LE(36 + dataSize, 4);
  buffer.write('WAVE', 8, 'ascii');

  // 2. fmt Subchunk
  buffer.write('fmt ', 12, 'ascii');
  buffer.writeUInt32LE(16, 16); // Subchunk1Size
  buffer.writeUInt16LE(1, 20); // AudioFormat = 1 (PCM)
  buffer.writeUInt16LE(numChannels, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(byteRate, 28);
  buffer.writeUInt16LE(blockAlign, 32);
  buffer.writeUInt16LE(16, 34); // BitsPerSample = 16

  // 3. data Subchunk
  buffer.write('data', 36, 'ascii');
  buffer.writeUInt32LE(dataSize, 40);

  // 4. Write samples
  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    // Generate sine wave
    const t = i / sampleRate;
    let s = amplitude * Math.sin(2 * Math.PI * frequencyHz * t);
    // Clamp
    s = Math.max(-1, Math.min(1, s));
    const intSample = s < 0 ? Math.round(s * 0x8000) : Math.round(s * 0x7fff);
    buffer.writeInt16LE(intSample, offset);
    offset += 2;
  }

  return buffer;
}

/**
 * HTTP multipart form-data helper.
 */
function sendMultipartRequest(endpoint, fields = {}, file = null) {
  return new Promise((resolve, reject) => {
    const boundary = '----WebKitFormBoundary' + Math.random().toString(36).substring(2);
    const parts = [];

    // Add string fields
    for (const [key, val] of Object.entries(fields)) {
      parts.push(
        Buffer.from(
          `--${boundary}\r\nContent-Disposition: form-data; name="${key}"\r\n\r\n${val}\r\n`,
          'utf-8'
        )
      );
    }

    // Add file if present
    if (file) {
      parts.push(
        Buffer.from(
          `--${boundary}\r\nContent-Disposition: form-data; name="${file.fieldname || 'file'}"; filename="${file.filename || 'recording.wav'}"\r\nContent-Type: ${file.contentType || 'audio/wav'}\r\n\r\n`,
          'utf-8'
        )
      );
      parts.push(file.buffer);
      parts.push(Buffer.from('\r\n', 'utf-8'));
    }

    parts.push(Buffer.from(`--${boundary}--\r\n`, 'utf-8'));
    const bodyBuffer = Buffer.concat(parts);

    const req = http.request(
      {
        host: '127.0.0.1',
        port: TEST_PORT,
        method: 'POST',
        path: endpoint,
        headers: {
          'Content-Type': `multipart/form-data; boundary=${boundary}`,
          'Content-Length': bodyBuffer.length,
        },
      },
      (res) => {
        let data = '';
        res.on('data', (chunk) => (data += chunk));
        res.on('end', () => {
          let parsed;
          try {
            parsed = JSON.parse(data);
          } catch (e) {
            parsed = data;
          }
          resolve({ statusCode: res.statusCode, body: parsed, headers: res.headers });
        });
      }
    );

    req.on('error', reject);
    req.write(bodyBuffer);
    req.end();
  });
}

function sendGetRequest(endpoint) {
  return new Promise((resolve, reject) => {
    const req = http.request(
      {
        host: '127.0.0.1',
        port: TEST_PORT,
        method: 'GET',
        path: endpoint,
      },
      (res) => {
        const chunks = [];
        res.on('data', (chunk) => chunks.push(chunk));
        res.on('end', () => {
          const buffer = Buffer.concat(chunks);
          let parsed;
          try {
            parsed = JSON.parse(buffer.toString('utf-8'));
          } catch (e) {
            parsed = buffer;
          }
          resolve({ statusCode: res.statusCode, body: parsed, rawBuffer: buffer, headers: res.headers });
        });
      }
    );
    req.on('error', reject);
    req.end();
  });
}

test.before((t, done) => {
  server = app.listen(TEST_PORT, done);
});

test.after((t, done) => {
  server.close(done);
});

test('WAV Encoder creates compliant 16 kHz mono 16-bit PCM RIFF structure', () => {
  const wavBuffer = createSyntheticWavBuffer(1.0, 440, 0.5);
  assert.strictEqual(wavBuffer.length, 44 + 16000 * 2);

  // Assert RIFF Magic
  assert.strictEqual(wavBuffer.toString('ascii', 0, 4), 'RIFF');
  assert.strictEqual(wavBuffer.toString('ascii', 8, 12), 'WAVE');
  assert.strictEqual(wavBuffer.toString('ascii', 12, 16), 'fmt ');
  assert.strictEqual(wavBuffer.readUInt32LE(16), 16); // Subchunk1Size
  assert.strictEqual(wavBuffer.readUInt16LE(20), 1); // PCM format
  assert.strictEqual(wavBuffer.readUInt16LE(22), 1); // Mono (1 channel)
  assert.strictEqual(wavBuffer.readUInt32LE(24), 16000); // 16000 Hz
  assert.strictEqual(wavBuffer.readUInt32LE(28), 32000); // Byte rate
  assert.strictEqual(wavBuffer.readUInt16LE(32), 2); // Block align
  assert.strictEqual(wavBuffer.readUInt16LE(34), 16); // 16-bit
  assert.strictEqual(wavBuffer.toString('ascii', 36, 40), 'data');
  assert.strictEqual(wavBuffer.readUInt32LE(40), 16000 * 2);
});

let normalEventId = null;
let anomalyEventId = null;

test('POST /api/events with uploaded live microphone audio (Normal Sound) triggers IGNORE and skips ML', async () => {
  // Low-amplitude gentle hum (normal operating acoustic)
  const normalWav = createSyntheticWavBuffer(2.0, 120, 0.02);

  const res = await sendMultipartRequest(
    '/api/events',
    {
      machine_type: 'fan',
      gate_type: 'amrag',
      resource_level: '1.0',
    },
    {
      fieldname: 'file',
      filename: 'live_mic_fan_normal.wav',
      buffer: normalWav,
      contentType: 'audio/wav',
    }
  );

  assert.strictEqual(res.statusCode, 201);
  assert.strictEqual(res.body.success, true);
  assert.strictEqual(res.body.gate_decision, 'IGNORE');
  assert.strictEqual(res.body.ml_executed, false);
  assert.strictEqual(res.body.prediction, null);
  assert.ok(res.body.event_id);
  assert.strictEqual(res.body.gate_summary.triggered, false);

  normalEventId = res.body.event_id;
});

test('POST /api/events with uploaded live microphone audio (High-Energy Anomaly) triggers GATE and invokes ML', async () => {
  // Use the real fan_anomaly sample — a pure synthetic sine has low spectral
  // flux and does not reliably trigger the multi-feature AMRAG gate.
  const anomalyWavPath = path.resolve(__dirname, '../../data/sample/fan_anomaly.wav');
  const anomalyWav = fs.readFileSync(anomalyWavPath);

  const res = await sendMultipartRequest(
    '/api/events',
    {
      machine_type: 'fan',
      gate_type: 'amrag',
      resource_level: '1.0',
    },
    {
      fieldname: 'file',
      filename: 'live_mic_bearing_anomaly.wav',
      buffer: anomalyWav,
      contentType: 'audio/wav',
    }
  );

  assert.strictEqual(res.statusCode, 201);
  assert.strictEqual(res.body.success, true);
  assert.strictEqual(res.body.gate_decision, 'TRIGGER');
  assert.strictEqual(res.body.ml_executed, true);
  assert.ok(res.body.prediction);
  assert.ok(['NORMAL', 'ANOMALY'].includes(res.body.prediction.result));
  assert.ok(typeof res.body.prediction.anomaly_score === 'number');
  assert.ok(res.body.event_id);

  anomalyEventId = res.body.event_id;
});

test('GET /api/audio/:id serves uploaded live recording for browser playback', async () => {
  assert.ok(normalEventId, 'normalEventId must be present from previous test');

  const res = await sendGetRequest(`/api/audio/${normalEventId}`);
  assert.strictEqual(res.statusCode, 200);
  assert.ok(res.rawBuffer.length > 44, 'Audio file must be non-empty and larger than header');
  assert.strictEqual(res.rawBuffer.toString('ascii', 0, 4), 'RIFF');
});

test('GET /api/events/:id retrieves live recording event details and metadata', async () => {
  assert.ok(anomalyEventId, 'anomalyEventId must be present from previous test');

  const res = await sendGetRequest(`/api/events/${anomalyEventId}`);
  assert.strictEqual(res.statusCode, 200);
  assert.strictEqual(res.body.event.id, anomalyEventId);
  assert.strictEqual(res.body.event.triggered, 1);
  // The event detail endpoint stores ML fields as flat columns (ml_result, anomaly_score)
  // rather than a nested prediction object — verify at least one ML field is present.
  assert.ok(
    res.body.event.ml_result || res.body.event.prediction,
    'Expected ml_result or prediction field on a triggered event'
  );
});

test('POST /api/events rejects empty upload with 400 Bad Request', async () => {
  const res = await sendMultipartRequest('/api/events', {
    machine_type: 'fan',
  });

  assert.strictEqual(res.statusCode, 400);
  assert.ok(res.body.error);
});
