/**
 * Backend API Integration Test Suite.
 */

const test = require('node:test');
const assert = require('node:assert');
const path = require('path');
const request = require('http');
const app = require('../src/server');

let server;
const TEST_PORT = 5099;

function makeRequest(method, endpoint, body = null, headers = {}) {
  return new Promise((resolve, reject) => {
    const req = request.request(
      {
        host: '127.0.0.1',
        port: TEST_PORT,
        method,
        path: endpoint,
        headers: {
          'Content-Type': 'application/json',
          ...headers,
        },
      },
      (res) => {
        let data = '';
        res.on('data', (chunk) => {
          data += chunk;
        });
        res.on('end', () => {
          try {
            resolve({
              statusCode: res.statusCode,
              body: JSON.parse(data),
            });
          } catch (e) {
            resolve({
              statusCode: res.statusCode,
              body: data,
            });
          }
        });
      }
    );

    req.on('error', reject);
    if (body) {
      req.write(typeof body === 'string' ? body : JSON.stringify(body));
    }
    req.end();
  });
}

test.before((t, done) => {
  server = app.listen(TEST_PORT, done);
});

test.after((t, done) => {
  server.close(done);
});

test('GET /api/health returns 200 OK', async () => {
  const res = await makeRequest('GET', '/api/health');
  assert.strictEqual(res.statusCode, 200);
  assert.strictEqual(res.body.status, 'healthy');
});

test('POST /api/events on normal audio triggers IGNORE and skips ML', async () => {
  const normalPath = path.resolve(__dirname, '../../data/sample/fan_normal.wav');
  const res = await makeRequest('POST', '/api/events', {
    audio_path: normalPath,
    machine_type: 'fan',
    gate_type: 'amrag',
    resource_level: 1.0,
  });

  assert.strictEqual(res.statusCode, 201);
  assert.strictEqual(res.body.gate_decision, 'IGNORE');
  assert.strictEqual(res.body.ml_executed, false);
});

test('POST /api/events on anomaly audio triggers GATE and executes ML', async () => {
  const anomalyPath = path.resolve(__dirname, '../../data/sample/fan_anomaly.wav');
  const res = await makeRequest('POST', '/api/events', {
    audio_path: anomalyPath,
    machine_type: 'fan',
    gate_type: 'amrag',
    resource_level: 1.0,
  });

  assert.strictEqual(res.statusCode, 201);
  assert.strictEqual(res.body.gate_decision, 'TRIGGER');
  assert.strictEqual(res.body.ml_executed, true);
  assert.ok(res.body.prediction);
});

test('GET /api/stats reports accurate counts and ML savings', async () => {
  const res = await makeRequest('GET', '/api/stats');
  assert.strictEqual(res.statusCode, 200);
  assert.ok(res.body.summary.total_recordings >= 2);
  assert.ok(res.body.summary.ml_processing_avoided_pct > 0);
});

test('GET /api/experiments returns benchmark comparison and ablation data', async () => {
  const res = await makeRequest('GET', '/api/experiments');
  assert.strictEqual(res.statusCode, 200);
  assert.ok(res.body.gate_comparison);
  assert.ok(res.body.ablation_study);
});
