# TASKS — Sparky Roadmap

Ordered, executable task list derived from
[`PROJECT_MISSION.md`](PROJECT_MISSION.md) and
[`ARCHITECTURE.md`](ARCHITECTURE.md).

**Design decision:** every deployment talks to an OpenAI-compatible LLM server
over HTTP. Jetson runs `llama.cpp server` on loopback; the Pi uses a LAN/cloud
server. Only the URL, model name, and credentials differ. The app has no
in-process LLM, no backend switch, and no local fallback. The in-process path
is kept only until the remote vision gate (A4.4) passes.

**How to use this file**
- Work top to bottom within a track. Complete a task only when its acceptance
  criteria are met.
- Tracks are independently shippable: **A** (core LLM portability) ships before
  **B** (product skills) and **C** (optional navigation).
- Tasks marked **GATE** block dependent work until resolved.
- Update checkboxes and add a date/note when a task completes.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked

---

## Track A — Core LLM Portability (ship first)

### A0. Documentation
- [x] **A0.1** Rewrite `PROJECT_MISSION.md` for the Pi-client + Jetson-standalone
  model, expanded skills, and privacy principles.
- [x] **A0.2** Create `ARCHITECTURE.md` with module layout, config, provider,
  data flows, server lifecycle, and SLOs.
- [x] **A0.3** Create this `TASKS.md`.
- [x] **A0.4** Update all three docs for the single-provider (HTTP-only) design,
  the vision gate, and server lifecycle.
- [x] **A0.5a** Apply the review corrections (gate ordering, Jetson asset set,
  credentials, TLS/Ollama notes, SLO assumptions, flag verification).
- [x] **A0.5b** Add a prominent "README is being rewritten" banner to
  `README.md` pointing to the new docs (full rewrite in A13).
  _(2026-11-24)_

**Acceptance:** docs exist, agree with each other, and reflect the HTTP-only
design with no backend switch.

### A1. Baseline and guardrails
- [x] **A1.1** Run the existing test suite and record the result.
  _(2026-11-24: 46 tests OK in 0.535 s; see `docs/baseline.md`.)_
- [x] **A1.2** Record baseline latency and memory on the dev Mac for a chat
  turn and a vision turn using the current in-process path.
  _(2026-11-24: `qwen35-2b` + `qwen35-4b`; harness in
  `scripts/benchmark_local.py`; results in `docs/baseline.md` and
  `docs/baseline.json`.)_
- [x] **A1.3** Note in the commit what behavior must not regress.
  _(2026-11-24: checklist in `docs/baseline.md`.)_

**Acceptance:** a baseline exists so the provider refactor can be proven
behavior-preserving, including HTTP overhead comparison later (A10.4/A11.2).

### A2. Runtime configuration (no backend switch)
- [x] **A2.1** Add `sparky/config/runtime.py` with a `RuntimeConfig` dataclass
  resolved as CLI → env → default. No profiles and no backend switch.
  _(2026-11-25: plus `--remote-*` CLI flags.)_
- [x] **A2.2** Make `ROBOT_REMOTE_URL` (default `http://127.0.0.1:8080/v1`) and
  `ROBOT_REMOTE_MODEL` the LLM location settings. If the model is unset and the
  server advertises exactly one at `/v1/models`, auto-detect it.
  _(2026-11-25.)_
- [x] **A2.3** Add `ROBOT_REMOTE_API_KEY`, `ROBOT_REMOTE_TIMEOUT`,
  `ROBOT_REMOTE_MAX_RETRIES`, `ROBOT_REMOTE_VERIFY_TLS`.
  _(2026-11-25.)_
- [x] **A2.4** Require an API key and HTTPS for any non-loopback URL; document
  installing a self-signed CA or using a reverse proxy.
  _(2026-11-25: documented in README.)_
- [x] **A2.5** Remove `ROBOT_LLM_BACKEND`, `ROBOT_LLM_GPU_LAYERS`, and
  `ROBOT_LLM_THREADS` from app settings; document them as server flags.
  _(2026-11-25: `ROBOT_LLM_BACKEND` does not exist; GPU/thread flags documented
  as server flags and kept in settings for the legacy loader until A4.6.)_
- [x] **A2.6** Fail fast on invalid/missing settings; never log secrets.
  _(2026-11-25: `ConfigError`, `repr=False` API key, redacted view.)_
- [x] **A2.7** Unit tests for precedence, defaults, and validation.
  _(2026-11-25: 17 new tests in `tests/test_runtime_config.py`.)_
- [x] **A2.8** Document the client-assets vs server-assets split in
  `download_models.py` help/README.
  _(2026-11-25: docs only; no downloader behavior change.)_

**Acceptance:** `RuntimeConfig` fully determines behavior; bad config produces
a clear startup error; Pi provisions client assets only and Jetson provisions
both sets.

### A3. Provider interface
- [ ] **A3.1** Add `sparky/llm/backends/base.py` with `LLMProvider`,
  `Capabilities`, and `HealthStatus`.
- [ ] **A3.2** Add `sparky/llm/backends/factory.py` returning the single
  provider; keep the interface as the extension seam.
- [ ] **A3.3** Document the message shape (OpenAI chat format, including
  `image_url` data URLs).

**Acceptance:** interface is type-annotated and matches `ARCHITECTURE.md` §6.

### A4. Remote adoption and vision gate (BLOCKING GATE)
- [ ] **A4.1** Add `RemoteHTTPProvider` alongside the existing local path.
- [ ] **A4.2** Build llama.cpp with `GGML_CUDA=on` on the Jetson; pin and record
  the version/commit and GGUF/mmproj hashes. Verify the pinned version's flags
  (`--mmproj`, `--jinja`, `--parallel`, prompt cache, mmproj GPU placement).
- [ ] **A4.3** Add `scripts/start-llm-server.sh` with the verified flags; launch
  the server locally and confirm `/health` and `/v1/models` respond.
- [ ] **A4.4 (GATE)** Prove chat, streaming, and **vision** against that server
  with the real GGUF + mmproj. Abort criteria:
  - `qwen35-2b` and `qwen35-4b` pass chat + streaming + vision → proceed.
  - Text works, vision fails → keep the in-process vision path and revisit the
    server choice before deleting anything.
  - Streaming fails but non-streaming works → proceed with non-streaming and
    record the latency impact.
  If the gate fails, stop and do not delete the local path.
- [ ] **A4.5** Wire callers behind `LLMProvider` and rewrite
  `test_conversation.py` / `test_vision.py` mocks to the provider interface.
- [ ] **A4.6** After the gate passes: delete `LocalGGUFProvider`,
  `llama-cpp-python`, and the in-process loader code.
- [ ] **A4.7** Confirm existing tests pass and generated outputs match the
  baseline; compare latency separately in A10.4/A11.2.

**Acceptance:** remote path is proven end-to-end (vision included) before any
local code is removed; rollback is trivial until A4.6.

### A5. Server hardening and lifecycle
- [ ] **A5.1** Add `scripts/sparky.service` (systemd) with `Restart=on-failure`
  and correct start ordering.
- [ ] **A5.2** Bind to loopback on the Jetson; require API key + TLS for any
  non-loopback LAN bind.
- [ ] **A5.3** Implement readiness polling (`/health`, 503 while loading), a
  bounded startup wait, and a warm-up text + image request after startup.
- [ ] **A5.4** Add crash auto-restart behavior, log rotation, and disable
  request-content/prompt logging (record the pinned version's log flags).
- [ ] **A5.5** Benchmark and tune server flags for latency; record chosen values
  in `ARCHITECTURE.md` and align `Conversation.context_tokens` with `-c`.

**Acceptance:** server starts reliably, becomes ready within the SLO, survives
crashes, and logs no content.

### A6. Contract and compatibility tests
- [ ] **A6.1** Add mock-HTTP tests for CI: success, SSE streaming, 401, 500,
  503-then-success, connection-refused-then-success, timeout, malformed SSE,
  split multi-byte UTF-8, `reasoning_content`, `[DONE]`, `/v1/models` parsing,
  and non-streaming transport fallback.
- [ ] **A6.2** Assert in tests that no message/image content is logged.
- [ ] **A6.3** Add a real llama.cpp server contract test on the Jetson (manual
  or nightly, not CI): chat, streaming, and a real image.
- [ ] **A6.4** Add optional Ollama compatibility checks.
- [ ] **A6.5** Record the compatibility matrix results in `ARCHITECTURE.md`.

**Acceptance:** CI covers the HTTP client; hardware contract test covers the
real server; matrix is documented.

### A7. Wire callers and CLI semantics
- [ ] **A7.1** Update `sparky/llm/text_llm.py` and
  `sparky/llm/vision_llm.py` to use the provider.
- [ ] **A7.2** Update `offline_robot.py` to build the provider from
  `RuntimeConfig` and poll server readiness at startup.
- [ ] **A7.3** Update `sparky/vision/turn.py` and `sparky/vision/check.py`.
- [ ] **A7.4** Make `--list-models` query `/v1/models` (OpenAI shape
  `{"data": [{"id": ...}]}`) and fail gracefully when the server is offline;
  keep `download_models.py --list-llms` for server provisioning inventory.
- [ ] **A7.5** Make `--check-vision` an HTTP probe with three outcomes:
  server unavailable / vision unsupported / vision OK (with model name).

**Acceptance:** `--look`, `--check-vision`, `--list-models`, and the main loop
work against the single provider.

### A8. Responsiveness and face states
- [ ] **A8.1** Stream tokens and flush completed sentences to `speak()`.
- [ ] **A8.2** Add a display thread with states `idle`, `listening`, `thinking`,
  `waking`, `speaking`, `error`; `waking` is a substate of `thinking`.
- [ ] **A8.3** Ensure the face animates independently of inference and server
  startup.
- [ ] **A8.4** Keep on-screen text and spoken text separate where needed
  (longer on screen).

**Acceptance:** audio begins after the first sentence; the face never freezes
while the model or server is loading.

### A9. Degraded mode
- [ ] **A9.1** Implement friendly messages for unreachable/loading/4xx/5xx /
  timeout / interrupted stream.
- [ ] **A9.2** Add tests for network partition, slow server, and server
  restart during a turn.
- [ ] **A9.3** Confirm local ASR/TTS/display/audio continue during an outage.
- [ ] **A9.4** Confirm no local model is loaded as fallback.

**Acceptance:** the robot remains usable and honest during any server failure.

### A10. Pi 5 validation
- [ ] **A10.1** Measure Whisper + Piper latency and memory on the Pi 5.
- [ ] **A10.2** Verify camera → HTTP VLM and `--check-vision`.
- [ ] **A10.3** Verify graceful behavior on Wi-Fi drop and server restart.
- [ ] **A10.4** Measure HTTP/SSE + base64 image overhead versus the A1
  in-process baseline; document results and the server class used.
- [ ] **A10.5** Confirm 8 GB headroom with the chosen UI renderer; document that
  4 GB is viable only for OS Lite + pygame + no navigation.
- [ ] **A10.6** Document LAN server setup (llama.cpp server / Ollama).

**Acceptance:** SLOs in `ARCHITECTURE.md` §12 are met or deviations documented.

### A11. Jetson validation
- [ ] **A11.1** Verify the llama.cpp CUDA build and server launch from A4.2/A4.3.
- [ ] **A11.2** Run `qwen35-2b` and `qwen35-4b` with mmproj; measure tokens/s and
  HTTP overhead.
- [ ] **A11.3** Verify `--check-vision` and a full offline chat/vision turn.
- [ ] **A11.4** Verify server cold start, readiness, warm-up, and crash restart.
- [ ] **A11.5** Document JetPack/CUDA/llama.cpp versions, cooling, and power.

**Acceptance:** fully offline operation at target speeds with a reproducible
server build.

### A12. Privacy and security review
- [ ] **A12.1** Document cloud vs LAN data flow and obtain explicit opt-in.
- [ ] **A12.2** Verify no content logging in client **and** server.
- [ ] **A12.3** Verify loopback default, API key, and TLS for non-loopback.
- [ ] **A12.4** Add content-safety filtering notes for child-facing output.

**Acceptance:** a written privacy note exists; defaults do not send data
off-device.

### A13. Documentation finalization
- [ ] **A13.1** Rewrite the README for both deployments, the server setup, the
  HTTP-only config, and the `ROBOT_REMOTE_*` naming (HTTP including loopback).
- [ ] **A13.2** Add troubleshooting: server unreachable, 503 while loading,
  wrong model name, TLS/CA errors, vision unsupported, port conflicts.
- [ ] **A13.3** Update `download_models.py` docs: server assets vs Pi client
  assets.

**Acceptance:** a new user can set up either deployment from the README alone.

---

## Track B — Product Skills (discovery, then build)

### B1. Decision gate
- [ ] **B1.1** Choose the UI stack: pygame/SDL on framebuffer, Qt, or browser
  kiosk. Record memory impact.
- [ ] **B1.2** Choose the screen: HDMI vs SPI/I²C and resolution.
- [ ] **B1.3** Choose piano input: USB-MIDI keyboard (recommended) vs
  microphone-based transcription.
- [ ] **B1.4** Choose call transport: LAN intercom vs internet VoIP vs
  cellular. Record privacy implications.
- [ ] **B1.5** Decide whether 4 GB or 8 GB Pi 5 is the minimum supported
  (subject to A10.5).

**Acceptance:** each choice is recorded in `ARCHITECTURE.md` before dependent
work starts.

### B2. Spikes (time-boxed proofs)
- [ ] **B2.1** MIDI note/rhythm detection spike.
- [ ] **B2.2** Pitch/onset detection spike for singing.
- [ ] **B2.3** Echo-cancellation spike for calls.
- [ ] **B2.4** Face renderer spike (one expression cycle + idle blink).
- [ ] **B2.5** One call-transport spike (connect, audio both ways, hang up).

**Acceptance:** each spike proves feasibility with numbers or documents why it
is deferred.

### B3. Skill architecture
- [ ] **B3.1** Add `sparky/skills/` with a skill base class and intent router.
- [ ] **B3.2** Add a priority-aware event bus (calls preempt lessons/stories).
- [ ] **B3.3** Refactor `offline_robot.py` to dispatch through the router.
- [ ] **B3.4** Keep chat, math, and vision as the first skills.

**Acceptance:** adding a new skill requires no edits to the main loop.

### B4. Piano tutor
- [ ] **B4.1** Add local audio/MIDI input and note detection.
- [ ] **B4.2** Add on-screen sheet/step display and instant feedback.
- [ ] **B4.3** Add practice modes and positive reinforcement.
- [ ] **B4.4** Keep all timing-critical logic local.

**Acceptance:** a kid can play a short exercise and receive correct
note/rhythm feedback without network dependency.

### B5. Singing tutor
- [ ] **B5.1** Add pitch detection against a local reference melody.
- [ ] **B5.2** Add sing-along and pitch/rhythm feedback.
- [ ] **B5.3** Render lyrics on screen with line highlighting.

**Acceptance:** feedback is local and responsive.

### B6. Story mode
- [ ] **B6.1** Add a local story/text library.
- [ ] **B6.2** Add long-form streaming TTS (not capped at 96 tokens).
- [ ] **B6.3** Render story text on screen synced to speech.
- [ ] **B6.4** Add "read again" / "next page" controls.

**Acceptance:** a story plays start to finish with matching on-screen text.

### B7. Family calling
- [ ] **B7.1** Implement the chosen transport with encryption.
- [ ] **B7.2** Add an approved-contacts allowlist and parental controls.
- [ ] **B7.3** Add call preemption and clear call-state UI.
- [ ] **B7.4** Ensure no call audio is recorded.
- [ ] **B7.5** Update the privacy section of the mission.

**Acceptance:** a child can call only approved contacts, and a call interrupts
other skills cleanly.

### B8. Content licensing
- [ ] **B8.1** Source public-domain or original stories, lyrics, and MIDI.
- [ ] **B8.2** Record licenses for every bundled asset.

**Acceptance:** no copyrighted content is bundled without a recorded license.

---

## Track C — Navigation (conditional)

- [ ] **C1.1** Add a motor HAT abstraction with odometry and an independent
  emergency stop.
- [ ] **C1.2** Add a simple local planner as a prioritized skill.
- [ ] **C1.3** Clarify in docs that the HAT handles actuation; planning stays on
  the main computer.
- [ ] **C1.4** Revisit perception acceleration only if real-time vision is
  required.

**Acceptance:** the robot can navigate a simple path and stop safely without
network dependency.

---

## Diagnostics & Release

- [ ] **D1.1** Add `offline_robot.py --status` showing effective config
  (secrets redacted), server health, model/vision capability, mic/camera, and
  last known latencies.
- [ ] **D2.1** Run the full test suite plus hardware smoke tests.
- [ ] **D2.2** Tag a release and archive baseline/benchmark results.

**Acceptance:** `--status` gives a one-screen health report; release notes list
known limitations.

---

## Current Recommended First Steps

1. A0.5b (README banner + pointer)
2. A1 (baseline, including in-process latency for later comparison)
3. A2 (config, no backend switch)
4. A3 (provider interface)
5. A4.1 → A4.2 → A4.3 (remote provider + pinned server build/launch)
6. **A4.4 GATE** (chat + streaming + vision with real GGUF/mmproj)
7. A5 (server hardening) + A6 (contract tests)
8. A4.5 → A4.6 → A4.7 (wire callers, then remove local path)
9. A7 → A13 (wiring, responsiveness, validation, docs)

Track B1 should run **in parallel** because its decisions affect A8 (UI) and
B2–B7.
