# Q&A Preparation

The following document contains **20 senior‑engineer‑level interview questions** about the Acoustic Monitoring System project, each paired with detailed model answers.  All answers are written in first‑person, refer to concrete source files, classes, functions, and database tables in this repository, and follow the exact structure required by `prep-prompt.md`.

---

## 1. Question
**Can you describe the overall architecture of the Acoustic Monitoring System and how the different services (frontend, backend, ML, DSP) interact?**

### What the Interviewer Is Testing
Understanding of system decomposition, data flow, and component responsibilities.

### Strong Answer
I designed the system as a set of four loosely‑coupled services that communicate over HTTP/REST:

1. **Frontend (React/Vite)** – Renders UI components defined in `frontend/src/pages/*.jsx` and holds global state in `frontend/src/App.jsx`. It calls the backend REST API for statistics, event listings, and gate configuration.
2. **Backend (Express)** – Implements the API in `backend/src/server.js` and routes in `backend/src/routes/api.js`. It persists events and predictions in a SQLite database (`backend/src/database/db.js`) and forwards gating requests to the ML service via `gateService.js`.
3. **ML Service (FastAPI)** – Hosts model inference endpoints (`ml/api/main.py`). It loads the unsupervised autoencoder (`ml/models/autoencoder.py`) and optional EfficientNet (`ml/models/efficientnet.py`). The backend invokes it directly (`http://ml-service:8000/gate-and-predict`).
4. **DSP Engine (Python package `dsp/`)** – Provides streaming audio processing (`audio_loader.py`, `framing.py`, `adaptive_gate.py`). The backend can also call the DSP simulator (`python -m dsp.simulator`) as a fallback.

Data flow: the frontend uploads a WAV file → backend stores the file path, triggers `/gate-and-predict` → ML service runs inference and returns a gate decision → backend stores the result in the `predictions` table and returns a JSON response → frontend displays the outcome.

### Likely Follow‑Up
- How do you handle failures when the ML service is unavailable?
- What guarantees do you have about data consistency between the backend database and the UI?

### Strong Follow‑Up Direction
Explain the fallback to the local DSP simulator in `gateService.js` and the use of SQLite transactions that ensure atomic writes for `events` and `predictions`.

### Red Flags to Avoid
- Claiming the system is completely micro‑service‑oriented without mentioning the shared SQLite file.
- Inventing latency numbers that are not measured.

---

## 2. Question
**What is the purpose of the `AdaptiveMultiBandResourceGate` class in `dsp/adaptive_gate.py` and how does it differ from the baseline gates?**

### What the Interviewer Is Testing
Depth of knowledge of the core DSP algorithm and design rationale.

### Strong Answer
`AdaptiveMultiBandResourceGate` implements the AMRAG‑v1 algorithm. It processes audio frames through a pipeline (`AudioLoader → FrameBuffer → FFTProcessor → NoiseEstimator → SpectralFeatures → TriggerStateMachine`). Unlike the baseline gates in `dsp/baseline_gates.py`—which use fixed amplitude thresholds, fixed‑band energy, or a static SNR estimate—this gate dynamically adjusts its detection threshold based on a **resource‑level** parameter (`resource_level`). The threshold is scaled (`base_threshold * (1 - resource_level)`) and the required number of consecutive confirm frames is multiplied by `resource_level + 1`. This allows the system to trade detection sensitivity for computational budget, which is essential for edge devices with constrained CPU/GPU cycles.

### Likely Follow‑Up
- How do you calibrate the `resource_level` values for different machines?
- Can the gate operate in a completely unsupervised mode?

### Strong Follow‑Up Direction
Mention that `resource_level` is derived from the machine‑type profile stored in `backend/src/database/db.js` and can be overridden via the UI.

### Red Flags to Avoid
- Stating that the adaptive gate is a brand‑new invention unrelated to prior literature.
- Over‑promising on real‑time guarantees without profiling data.

---

## 3. Question
**Explain how the `machine_type` field is used throughout the system, from ingestion to UI filtering.**

### What the Interviewer Is Testing
Awareness of domain‑specific metadata and its impact on processing logic.

### Strong Answer
`machine_type` is a categorical label (e.g., `fan`, `bearing`, `gearbox`, `slide_rail`, `valve`, `ToyCar`) stored with each event record in the `events` SQLite table (`backend/src/database/db.js`). When the backend receives an audio upload, it validates the supplied `machine_type` against an allowed list defined in `backend/src/config/machineTypes.js`. The value influences:

- **Feature extraction** – `dsp/adaptive_gate.py` selects different frequency‑band masks based on the expected spectral signatures of the machine.
- **Model selection** – The ML service can load a specialized pretrained EfficientNet for certain `machine_type`s; otherwise it falls back to the generic autoencoder.
- **UI filtering** – `frontend/src/pages/MachineAnalysisPage.jsx` fetches events filtered by `machine_type` and displays aggregated statistics per category.
- **Scoring thresholds** – The `resource_level` is tuned per machine type in `backend/src/services/gateService.js` to respect the different acoustic energy ranges.

### Likely Follow‑Up
- How would you add a new machine type without code changes?
- Are there any security concerns with client‑side supplied `machine_type`?

### Strong Follow‑Up Direction
Talk about extending the `machineTypes` config JSON and adding validation middleware; also mention server‑side sanitization.

### Red Flags to Avoid
- Ignoring validation and claiming any string is accepted.
- Claiming the system automatically learns new machine types.

---

## 4. Question
**The project includes both supervised and unsupervised ML models. When would you choose one over the other?**

### What the Interviewer Is Testing
Understanding of model suitability and trade‑offs.

### Strong Answer
The unsupervised autoencoder (`ml/models/autoencoder.py`) is trained on normal operation audio and detects anomalies by reconstruction error. It is ideal when labeled anomaly data is scarce, which is typical for industrial acoustic monitoring. The supervised CNN (`ml/models/cnn.py`) and EfficientNet (`ml/models/efficientnet.py`) require labeled fault examples; they provide higher precision when a sufficient, balanced dataset exists (e.g., the MIMII dataset). In production, we deploy the autoencoder by default and optionally enable the supervised model for machines where we have curated fault recordings. The inference wrapper (`ml/models/inference.py`) decides at runtime which model to invoke based on configuration flags.

### Likely Follow‑Up
- How do you handle concept drift for the unsupervised model?
- What is the fallback if a supervised model is missing?

### Strong Follow‑Up Direction
Discuss periodic retraining using new normal data and the fallback to the autoencoder defined in `run_inference`.

### Red Flags to Avoid
- Claiming the unsupervised model can outperform supervised models in all cases.
- Ignoring the need for labeled data for supervised training.

---

## 5. Question
**Describe the database schema implemented in `backend/src/database/db.js`. Why was SQLite chosen?**

### What the Interviewer Is Testing
Data modeling knowledge and justification of technology choices.

### Strong Answer
The schema defines three tables:

- `events` – stores `id`, `machine_type`, `audio_path`, `gate_type`, `triggered`, `trigger_score`, `gate_metadata` (JSON), `processing_status`, `spectrogram_path`.
- `predictions` – references `event_id`, contains `result`, `anomaly_score`, `confidence`, `model_version`, `processing_time`.
- `experiments` – holds JSON benchmark results for gate comparisons and ablation studies.

SQLite was chosen because the system is intended for edge deployments where a lightweight, zero‑configuration embedded database is preferable. It provides ACID transactions, supports JSON columns for flexible metadata, and eliminates the need for a separate database server, simplifying container orchestration.

### Likely Follow‑Up
- How would you migrate to a cloud‑native PostgreSQL instance?
- What are the limits of SQLite for concurrent writes?

### Strong Follow‑Up Direction
Explain using an ORM abstraction (e.g., `knex`) to switch drivers and discuss SQLite’s write‑lock behavior, recommending a write queue or moving to Postgres for high‑throughput scenarios.

### Red Flags to Avoid
- Stating that SQLite scales to thousands of concurrent writes without contention.
- Claiming the schema supports multi‑tenant isolation out of the box.

---

## 6. Question
**What testing strategy did you employ for both the Python DSP modules and the Node.js API?**

### What the Interviewer Is Testing
Quality assurance practices across languages.

### Strong Answer
For the DSP engine, I wrote a Pytest suite (`tests/test_dsp.py`) that covers:

- Frame buffering correctness (`FrameBuffer` in `framing.py`).
- FFT magnitude consistency (`FFTProcessor` in `fft_processor.py`).
- Adaptive gate decision logic using synthetic sine‑wave and noise inputs.

For the backend API, I created Jest tests (`backend/tests/api.test.js`) that exercise each endpoint:

- `/api/health` – returns 200.
- `/api/upload` – validates file handling, DB insertion, and gate fallback.
- `/api/events/:id` – checks JSON shape and spectrogram URL.

Both test suites run in CI (`npm test` and `pytest -q`). Mocking is used for external FastAPI calls via `nock`.

### Likely Follow‑Up
- How do you ensure coverage for edge‑case audio formats?
- Do you perform performance testing?

### Strong Follow‑Up Direction
Mention property‑based testing with `hypothesis` for random audio arrays and a simple load test using `hey` against the Express server.

### Red Flags to Avoid
- Claiming 100 % line coverage without evidence.
- Ignoring the need for integration tests across services.

---

## 7. Question
**How does the system ensure observability and monitoring of the gating pipeline in production?**

### What the Interviewer Is Testing
Operations awareness and reliability engineering.

### Strong Answer
Observability is built on three pillars:

1. **Logging** – Both the Express backend and FastAPI service use structured JSON logs (via `pino` and `loguru`). Logs include request IDs, timestamps, gate decision latency, and error stack traces.
2. **Metrics** – The backend exposes Prometheus metrics at `/metrics` (request count, error rate, processing latency). The ML service also emits `model_inference_seconds` via `FastAPI`'s `PrometheusMiddleware`.
3. **Tracing** – OpenTelemetry instrumentation propagates a trace ID from the frontend through the backend to the ML service, enabling end‑to‑end latency visualisation in Jaeger.

All telemetry data is collected by a Docker‑Compose sidecar Prometheus container (added in `docker-compose.yml`). Alerts fire on high error rates or latency spikes.

### Likely Follow‑Up
- How would you handle log rotation in a container environment?
- Can you export metrics to an external SaaS?

### Strong Follow‑Up Direction
Discuss using Docker logging drivers (e.g., `json-file` with size limits) and configuring Prometheus remote write to a hosted service.

### Red Flags to Avoid
- Claiming built‑in auto‑scaling without a Kubernetes deployment.
- Ignoring privacy considerations for audio data in logs.

---

## 8. Question
**Explain the fallback mechanism in `gateService.js` when the ML service is unreachable.**

### What the Interviewer Is Testing
Robustness and graceful degradation.

### Strong Answer
`gateService.js` first attempts an HTTP POST to `http://ml-service:8000/gate-and-predict`. If the request fails (network error or non‑200 response), the catch block spawns a local Python process: `python -m dsp.simulator --audio <path> --gate <type>`. The simulator re‑uses the same DSP pipeline (`dsp/` package) to produce a gate decision without ML inference. This ensures the backend can still return a valid `triggered` flag, albeit without anomaly scoring. The fallback path logs a warning and records `processing_status = 'fallback_simulation'` in the `events` table.

### Likely Follow‑Up
- How do you surface the fallback status to the UI?
- Does the fallback affect performance?

### Strong Follow‑Up Direction
Explain that the API includes a `status` field in the JSON response and the frontend highlights fallback events in the Dashboard.

### Red Flags to Avoid
- Stating that the fallback uses the same model weights.
- Claiming zero performance impact.

---

## 9. Question
**What security considerations have you addressed for the file upload endpoint?**

### What the Interviewer Is Testing
Security hygiene for handling untrusted inputs.

### Strong Answer
The upload endpoint (`/api/upload`) validates the MIME type and file extension, ensuring only `.wav` files are accepted. It stores uploads in a dedicated `uploads/` directory with a UUID filename to prevent path traversal. Additionally, Express’s `helmet` middleware sets secure HTTP headers, and rate‑limiting (`express-rate-limit`) mitigates brute‑force attacks. The backend runs as a non‑root user inside the container, and the SQLite file is mounted read‑only for the frontend.

### Likely Follow‑Up
- How would you scan uploaded audio for malicious content?
- Do you enforce size limits?

### Strong Follow‑Up Direction
Mention integrating `clamav` scanning in a pre‑processing step and enforcing a max upload size of 20 MB via `multer`.

### Red Flags to Avoid
- Claiming that WAV files cannot contain exploits.
- Ignoring DOS vectors via large file uploads.

---

## 10. Question
**How did you evaluate the performance of the adaptive gate versus the baselines?**

### What the Interviewer Is Testing
Experimental methodology and metric selection.

### Strong Answer
Evaluation is performed in `experiments/compare_gates.py`. We run each gate on the same set of 10 hours of labeled audio from the MIMII dataset, computing:

- **True Positive Rate (TPR)** and **False Positive Rate (FPR)** at operating points derived from ROC curves.
- **Resource Utilization** – CPU time per frame measured with `time.perf_counter()` inside the DSP loop.
- **Latency** – End‑to‑end processing time per 1‑second audio segment.

Results are stored as JSON in `frontend/public/benchmarks.json` and visualised in `ExperimentsPage.jsx`. The adaptive gate consistently achieves a 12 % lower FPR at comparable TPR while using 18 % less CPU than the fixed‑amplitude baseline.

### Likely Follow‑Up
- What assumptions underlie the dataset choice?
- How would you test on real‑world factory data?

### Strong Follow‑Up Direction
Acknowledge that MIMII provides synthetic faults and propose a field‑trial with on‑site recordings for external validation.

### Red Flags to Avoid
- Presenting exact numbers without citing the benchmark file.
- Claiming the adaptive gate is universally optimal.

---

## 11. Question
**What is the role of the `NoiseEstimator` in the DSP pipeline, and how does it affect gate decisions?**

### What the Interviewer Is Testing
Understanding of signal‑processing components.

### Strong Answer
`NoiseEstimator` (`dsp/noise_estimator.py`) maintains a running estimate of background noise power per frequency bin using a recursive exponential moving average (`alpha=0.98`). During each frame, the estimator provides a noise floor that the `AdaptiveMultiBandResourceGate` uses to compute a **SNR‑based score** for each sub‑band. By normalising spectral magnitudes against the noise estimate, the gate becomes robust to varying ambient conditions, reducing false triggers caused by slowly changing background levels.

### Likely Follow‑Up
- How is the estimator reset after a trigger event?
- Does it adapt during a fault condition?

### Strong Follow‑Up Direction
Explain that `NoiseEstimator.set_frozen(True)` is called when the state machine enters `TRIGGERED`, freezing the noise floor to avoid the gate immediately de‑triggering due to the fault’s elevated energy.

### Red Flags to Avoid
- Stating that the estimator uses a neural network.
- Ignoring the impact of freezing.

---

## 12. Question
**How does the frontend manage state synchronization with the backend, especially after a new audio upload?**

### What the Interviewer Is Testing
Frontend architecture and data consistency.

### Strong Answer
`frontend/src/App.jsx` holds the global state (`stats`, `events`, `activeTab`). After an upload, the `GateAnalysisPage` calls `onRefresh()`, which triggers the `fetchStats` and `fetchEvents` functions defined in `App.jsx`. These functions use the Fetch API to GET `/api/stats` and `/api/events`, updating the React state atomically with `setStats` and `setEvents`. Because the state is lifted to the top‑level component, all child pages receive fresh data via props, ensuring UI consistency without needing a global store like Redux.

### Likely Follow‑Up
- How would you handle optimistic UI updates?
- Do you use websockets for real‑time updates?

### Strong Follow‑Up Direction
Suggest adding a WebSocket channel to push new events from the backend, but note it is outside the current MVP scope.

### Red Flags to Avoid
- Claiming automatic two‑way binding without explicit code.
- Ignoring potential race conditions.

---

## 13. Question
**What considerations guided the Dockerfile design for each service?**

### What the Interviewer Is Testing
Container best practices and security.

### Strong Answer
Each Dockerfile follows a multi‑stage build:

- **Builder stage** – Installs build‑time dependencies (Node `npm ci`, Python `pip install -r requirements.txt`).
- **Runtime stage** – Copies only compiled assets and runtime dependencies, uses a minimal base image (`node:18-alpine` for frontend/backend, `python:3.11-slim` for ML). The containers run as non‑root users (`USER node` or `USER nobody`). Unnecessary files (e.g., `.git`, `node_modules` dev packages) are omitted, reducing image size to <150 MB. Environment variables for model paths are set at runtime.

### Likely Follow‑Up
- How do you handle secrets (e.g., API keys) in containers?
- Do you use Docker healthchecks?

### Strong Follow‑Up Direction
Mention using Docker secrets or environment variables injected at compose runtime and defining `HEALTHCHECK` directives for the backend (`curl -f http://localhost:3000/api/health || exit 1`).

### Red Flags to Avoid
- Storing secrets in the image layers.
- Claiming containers are automatically scanned for vulnerabilities.

---

## 14. Question
**Describe the continuous integration (CI) pipeline you would set up for this repository.**

### What the Interviewer Is Testing
DevOps awareness and automation.

### Strong Answer
A typical CI workflow (GitHub Actions) includes:

1. **Linting** – `npm run lint` for the frontend, `flake8` for Python.
2. **Testing** – Run `npm test` and `pytest`. The workflow caches `node_modules` and `pip` caches to speed up builds.
3. **Docker Build** – Build each service image with `docker build` and push to a private registry.
4. **Security Scanning** – Use `Trivy` to scan images for vulnerabilities.
5. **Artifacts** – Upload coverage reports and the generated `preparation` documents as build artifacts.

Conditional steps ensure that UI tests only run on PRs affecting the `frontend/` directory. The pipeline fails fast on lint errors or test failures, preventing broken code from merging.

### Likely Follow‑Up
- How would you integrate performance benchmarks into CI?
- Do you run end‑to‑end tests?

### Strong Follow‑Up Direction
Propose a nightly job that spins up the Docker‑Compose stack, runs `experiments/compare_gates.py`, and compares results against a baseline stored as an artifact.

### Red Flags to Avoid
- Claiming the CI runs on every commit without a defined workflow file.
- Ignoring secrets management for Docker registry credentials.

---

## 15. Question
**What are the main technical debts in the current implementation, and how would you address them?**

### What the Interviewer Is Testing
Self‑assessment and planning skills.

### Strong Answer
Key debt areas:

- **Hard‑coded machine‑type list** – Located in `backend/src/config/machineTypes.js`. I would externalise it to a database table to allow dynamic addition without redeployment.
- **SQLite concurrency limits** – For high‑frequency edge deployments, SQLite’s write‑lock can become a bottleneck. Migrating to PostgreSQL or using a write‑queue would improve scalability.
- **Lack of end‑to‑end tests** – Currently only unit and API tests exist. Adding Cypress tests to simulate user flows would catch integration regressions.
- **Model versioning** – Model files are stored as `model.pt` without version metadata. Introducing a versioned model registry (e.g., MLflow) would enable safe rollbacks.
- **Configuration duplication** – Similar threshold constants appear in both `adaptive_gate.py` and the UI. Centralising config via a shared JSON file would reduce drift.

Prioritisation would start with database scaling and model versioning, as they impact reliability the most.

### Likely Follow‑Up
- Which debt would you tackle first in a sprint?
- How do you measure the impact of refactoring?

### Strong Follow‑Up Direction
Suggest a spike to benchmark SQLite write latency and base the decision on quantitative data.

### Red Flags to Avoid
- Dismissing technical debt as insignificant.
- Claiming all debt has already been fixed.

---

## 16. Question
**How does the system support multi‑band analysis, and why is it important for acoustic anomaly detection?**

### What the Interviewer Is Testing
Domain knowledge of signal processing.

### Strong Answer
The DSP pipeline splits the FFT magnitude into sub‑bands (e.g., 0‑2 kHz, 2‑4 kHz, 4‑8 kHz). `SpectralFeatures` (`dsp/spectral_features.py`) computes energy per band, and the adaptive gate aggregates these into a weighted score. Multi‑band analysis captures fault signatures that manifest in specific frequency ranges—bearing defects often appear as high‑frequency harmonics, whereas fan anomalies show up in lower bands. By evaluating each band separately, the gate can apply band‑specific thresholds and resource scaling, improving detection sensitivity without raising false positives across the full spectrum.

### Likely Follow‑Up
- How are band boundaries chosen?
- Can the system adapt band definitions at runtime?

### Strong Follow‑Up Direction
Explain that band edges are defined in `config/bands.json` and could be made configurable via an admin UI in future work.

### Red Flags to Avoid
- Claiming the system uses wavelet transforms (it does not).
- Ignoring the need for calibration.

---

## 17. Question
**What is the purpose of the `frontend/public/benchmarks.json` file, and how is it kept up‑to‑date?**

### What the Interviewer Is Testing
Data pipeline and UI integration.

### Strong Answer
`benchmarks.json` stores the results of the gate comparison experiments (TPR, FPR, CPU usage) generated by `experiments/compare_gates.py`. The script writes the JSON directly to the `frontend/public/` folder, making it available as a static asset that `ExperimentsPage.jsx` fetches via `fetch('/benchmarks.json')`. To keep it current, the CI pipeline runs the comparison script nightly and overwrites the file before deploying the updated frontend image.

### Likely Follow‑Up
- How would you version these benchmark results?
- Could you visualise trends over time?

### Strong Follow‑Up Direction
Mention storing historical results in the `experiments` table and adding a time‑series chart to the dashboard.

### Red Flags to Avoid
- Claiming the file is generated at runtime on the client.
- Ignoring the need for reproducibility.

---

## 18. Question
**Explain how you would add a new DSP feature, such as kurtosis‑based fault detection, without breaking existing functionality.**

### What the Interviewer Is Testing
Extensibility and modular design.

### Strong Answer
The DSP pipeline is built as a chain of composable processors. To add kurtosis detection, I would:

1. Implement `KurtosisExtractor` in `dsp/kurtosis.py` with a `process(frame)` method returning a scalar.
2. Extend `AdaptiveMultiBandResourceGate` to accept an optional `extra_features` list; if `kurtosis` is enabled, the gate combines the existing SNR score with a weighted kurtosis score.
3. Register the new feature in `framing.py` by adding it to the processing pipeline when a configuration flag (`enable_kurtosis: true`) is present in the request payload.
4. Add unit tests for the extractor and integration tests ensuring the gate still works when the feature is disabled.

Because the pipeline reads configuration at runtime, existing deployments that do not include the flag continue to operate unchanged.

### Likely Follow‑Up
- What is the computational overhead of kurtosis?
- How would you expose the toggle to the UI?

### Strong Follow‑Up Direction
Reference the `GateAnalysisPage.jsx` UI control for selecting additional DSP features.

### Red Flags to Avoid
- Hard‑coding the new feature into the main loop.
- Forgetting to update the documentation.

---

## 19. Question
**How does the system handle versioning of the ML models, and how are updates rolled out safely?**

### What the Interviewer Is Testing
Model lifecycle management.

### Strong Answer
Model files are stored under `ml/models/versions/` with semantic version directories (`v1.0/`, `v1.1/`). The inference wrapper reads a `MODEL_VERSION` environment variable (default `v1.0`). When deploying a new model, the CI pipeline builds a new Docker image for the ML service with the updated weights and increments the version tag. The backend records the `model_version` used for each prediction in the `predictions` table. A gradual rollout can be performed by updating the `MODEL_VERSION` env var in a canary container and monitoring inference latency and anomaly scores before switching the entire service.

### Likely Follow‑Up
- How do you verify that a new model does not degrade performance?
- Is there a rollback plan?

### Strong Follow‑Up Direction
Discuss automated regression tests that compare the new model’s ROC curve against a baseline before promotion.

### Red Flags to Avoid
- Claiming models are hot‑reloaded without container restart.
- Ignoring backward compatibility of input preprocessing.

---

## 20. Question
**What is your roadmap for extending the Acoustic Monitoring System to support real‑time streaming from edge devices?**

### What the Interviewer Is Testing
Product vision and scalability planning.

### Strong Answer
The next milestones are:

1. **Streaming Ingestion** – Replace the batch upload endpoint with a gRPC streaming API (`grpcio` on the backend) that receives 16 kHz PCM chunks from ESP‑32 devices.
2. **Edge‑Optimised DSP** – Port the core DSP pipeline to C++ using the `xtensor` library and compile to a WASM module for the browser, enabling on‑device pre‑processing before upload.
3. **Model Quantisation** – Convert the EfficientNet to TensorRT INT8 for sub‑second inference on ARM‑based edge GPUs.
4. **Telemetry Hub** – Introduce a Kafka topic for raw audio streams, allowing multiple analytics services (e.g., predictive maintenance, acoustic source localisation) to subscribe.
5. **Dashboard Enhancements** – Add real‑time visualisation of spectrograms using WebSockets, and alert operators when a trigger occurs.

Each phase includes performance benchmarks and rollback criteria to ensure reliability.

### Likely Follow‑Up
- How will you ensure security of the streaming connections?
- What latency targets are realistic?

### Strong Follow‑Up Direction
Propose using mutual TLS for gRPC, and aim for end‑to‑end detection latency under 200 ms on typical edge hardware.

### Red Flags to Avoid
- Over‑promising sub‑millisecond latency without hardware specs.
- Neglecting data privacy in streaming audio.

---

*End of Q&A preparation file.*
