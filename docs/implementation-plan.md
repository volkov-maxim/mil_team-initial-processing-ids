# Technical Implementation Plan (Phased Checklist)

This plan translates the high-level specification ([docs/high-level-spec.md](docs/high-level-spec.md)) into an executable development roadmap with architecture, concrete components, dependencies, and milestone gates.

## 1) Target Architecture (Implementation Blueprint)

- [ ] **Runtime topology**
  - [ ] Single Python API service exposing `POST /v1/process-document`.
  - [ ] Internal pipeline execution follows strict contract order:
    1. Validate input
    2. Align image
    3. OCR
    4. Extract fields
    5. Validate fields
    6. Optional external fallback
    7. Compose response
- [ ] **Artifact strategy**
  - [ ] Save aligned image + overlay image under request-scoped artifact directory.
  - [ ] Return artifact references as local path (MVP default) and allow future URL/base64 mode.
- [ ] **Execution modes**
  - [ ] Device selector supports `cpu`, `cuda`, and `auto`.
  - [ ] Hard guardrail for GPU memory budget target (`<= 10 GB`) through model/config selection.

## 2) Proposed Project Structure and Components

- [ ] Create package layout:

```text
src/
  app/
    main.py
    api/
      routes.py
      schemas.py
      errors.py
    core/
      config.py
      logging.py
      exceptions.py
    pipeline/
      processing.py
      context.py
      result.py
    preprocessing/
      image_io.py
      document_preprocessor.py
    ocr/
      detector.py
      recognizer.py
    extraction/
      normalizers.py
      base_extractor.py
      rules_common.py
      bank_card_extractor.py
      id_card_extractor.py
      drivers_license_extractor.py
      dispatcher.py
    validation/
      field_validators.py
      consistency_checks.py
      confidence.py
    fallback/
      client.py
      prompt_builder.py
      adapter.py
      policy.py
    storage/
      artifacts.py
    telemetry/
      metrics.py
      tracing.py
  tests/
    unit/
    integration/
    e2e/
Dockerfile
compose.yaml
.env.example
```

### 2.1 API layer (`src/app/api`)
- [ ] **Endpoint**
  - [ ] Implement `POST /v1/process-document` in `routes.py`.
- [ ] **Request schemas** (`schemas.py`)
  - [ ] `ProcessDocumentRequest` multipart contract (`image`, `document_type_hint`, `use_external_fallback`).
- [ ] **Response schemas** (`schemas.py`)
  - [ ] `ProcessDocumentResponse` with:
    - [ ] `request_id`
    - [ ] `document_type_detected`
    - [ ] `aligned_image`
    - [ ] `detections`
    - [ ] `fields`
    - [ ] `field_confidence`
    - [ ] `validation_flags`
    - [ ] `processing_metadata`
- [ ] **Error contract** (`errors.py`)
  - [ ] Typed error model for `400`, `422`, `500` with `error_code` and `message`.

### 2.2 Pipeline processing (`src/app/pipeline`)
- [ ] **Function: `process_document_pipeline`** (`processing.py`)
  - [ ] `process_document_pipeline(request_context) -> PipelineResult`
  - [ ] Enforces stage order and fail/partial semantics.
- [ ] **Class: `PipelineContext`** (`context.py`)
  - [ ] Holds request metadata, feature flags, intermediate artifacts.
- [ ] **Class: `PipelineResult`** (`result.py`)
  - [ ] Holds final payload + diagnostics + timing.

### 2.3 Preprocessing/alignment (`src/app/preprocessing`)
- [ ] **Class: `DocumentPreprocessor`** (`document_preprocessor.py`)
  - [ ] `validate_type_size_readability(image_bytes) -> ValidationOutcome`
  - [ ] `align(image) -> AlignmentResult`
  - [ ] Includes boundary detection + perspective correction + rotation normalization.
  - [ ] `denoise_contrast(image) -> image`

### 2.4 OCR (`src/app/ocr`)
- [ ] **Class: `TextDetector`** (`detector.py`)
  - [ ] `detect(aligned_image) -> list[TextRegion]`
- [ ] **Class: `TextRecognizer`** (`recognizer.py`)
  - [ ] `recognize(aligned_image, regions) -> list[RecognizedToken]`
  - [ ] `group_tokens_to_lines(tokens) -> list[RecognizedLine]`
  - [ ] Produces line-level output with confidence and boxes.

### 2.5 Field extraction (`src/app/extraction`)
- [ ] **Class: `DocumentTypeDispatcher`** (`dispatcher.py`)
  - [ ] `resolve_extractor(document_type_hint, ocr_lines) -> BaseExtractor`
- [ ] **Abstract class: `BaseExtractor`** (`base_extractor.py`)
  - [ ] `extract(ocr_lines) -> ExtractedFields`
- [ ] **Shared rule helpers** (`rules_common.py`)
  - [ ] Reusable regex/pattern helpers used by all extractors.
  - [ ] Label/synonym matching and token cleanup utilities.
  - [ ] Common extraction confidence heuristics shared across document types.
- [ ] **Concrete extractors**
  - [ ] `BankCardExtractor` (`bank_card_extractor.py`)
  - [ ] `IdCardExtractor` (`id_card_extractor.py`)
  - [ ] `DriversLicenseExtractor` (`drivers_license_extractor.py`)
- [ ] **Normalization utilities** (`normalizers.py`)
  - [ ] `normalize_date(...)`
  - [ ] `normalize_name(...)`
  - [ ] `normalize_document_number(...)`

### 2.6 Validation and confidence (`src/app/validation`)
- [ ] **Class: `FieldValidators`** (`field_validators.py`)
  - [ ] Date plausibility checks
  - [ ] Format/pattern checks (card number, license/document number)
- [ ] **Class: `ConsistencyChecks`** (`consistency_checks.py`)
  - [ ] Cross-field consistency flags
- [ ] **Class: `ConfidenceScorer`** (`confidence.py`)
  - [ ] Field-level confidence aggregation
- [ ] **Output**
  - [ ] `validation_flags`
  - [ ] `field_confidence`
  - [ ] Missing values represented as explicit `null`.

### 2.7 External fallback (`src/app/fallback`)
- [ ] **Class: `FallbackClient`** (`client.py`)
  - [ ] Configurable `base_url`/proxy/env-driven credentials.
- [ ] **Prompt builder** (`prompt_builder.py`)
  - [ ] Build structured fallback request payloads from OCR lines and extracted fields.
  - [ ] Include document type hint, validation flags, and confidence context in prompt input.
  - [ ] Keep prompt format consistent across document types and fallback providers.
- [ ] **Class: `FallbackAdapter`** (`adapter.py`)
  - [ ] Converts local OCR + extracted data into fallback request.
  - [ ] Merges fallback response with provenance flags.
- [ ] **Class: `FallbackPolicy`** (`policy.py`)
  - [ ] Invoke only when confidence below configured thresholds and request allows fallback.

### 2.8 Storage and observability
- [ ] **Artifact management** (`storage/artifacts.py`)
  - [ ] Create request folder, persist aligned + overlay images, return references.
  - [ ] `draw_detections(aligned_image, detections) -> annotated_image`
- [ ] **Class: `MetricsCollector`** (`telemetry/metrics.py`)
  - [ ] Per-stage latency and overall latency.
- [ ] **Class: `TraceContext`** (`telemetry/tracing.py`)
  - [ ] Device used (CPU/GPU), model versions, fallback usage.

## 3) API Contract Checklist (MVP)

- [ ] `POST /v1/process-document` accepts multipart image + optional params.
- [ ] Return `400` for invalid media/type/size violations.
- [ ] Return `422` for unreadable/non-document images or alignment failure.
- [ ] Return `500` for internal unhandled errors.
- [ ] On partial extraction, return success with explicit `null` fields and confidence/flags.

## 4) Phased Development Order (with Milestones and Dependencies)

## Phase 0 — Foundation & Scaffolding
- [ ] Initialize project skeleton, dependency management, and configuration model.
- [ ] Add structured logging and exception primitives.
- [ ] Add base Dockerfile + compose service.

**Dependencies:** none  
**Milestone M0 (Ready-to-code runtime):** app starts, health route works, config loads from env.

## Phase 1 — API Contract & Error Model
- [ ] Implement route, request parser, response envelopes, typed error handlers.
- [ ] Add request ID generation + propagation.
- [ ] Add integration test for request validation outcomes.

**Depends on:** Phase 0  
**Milestone M1 (Contract-first API):** endpoint returns valid placeholders and error codes per spec.

## Phase 2 — Input Validation + Alignment
- [ ] Implement image type/size/readability checks.
- [ ] Implement document boundary detection and perspective/rotation correction.
- [ ] Return typed diagnostics on alignment failure.

**Depends on:** Phase 1  
**Milestone M2 (Alignment stage complete):** aligned image artifact generated for representative samples.

## Phase 3 — OCR Detection + Recognition
- [ ] Integrate local OCR stack (detector + recognizer).
- [ ] Produce token/line outputs with bounding geometry and confidence.
- [ ] Implement overlay rendering on aligned image.

**Depends on:** Phase 2  
**Milestone M3 (OCR stage complete):** detections + text are returned with overlay artifact.

## Phase 4 — Schema Extraction & Normalization
- [ ] Implement document-type dispatcher.
- [ ] Implement extractor rules for bank card, ID card, and driver’s license.
- [ ] Implement date/name/number normalization rules and parse-status handling.

**Depends on:** Phase 3  
**Milestone M4 (Structured extraction complete):** required fields mapped for all 3 document classes.

## Phase 5 — Validation & Confidence Layer
- [ ] Add field plausibility and format checks.
- [ ] Add per-field confidence aggregation + validation flags.
- [ ] Ensure partial extraction contract (explicit nulls) is always respected.

**Depends on:** Phase 4  
**Milestone M5 (Trustworthy outputs):** response includes field confidence + validation flags.

## Phase 6 — Optional External Fallback
- [ ] Implement fallback client with configurable `base_url` and proxy support.
- [ ] Add threshold-driven invocation logic.
- [ ] Merge fallback fields with provenance metadata.

**Depends on:** Phase 5  
**Milestone M6 (Resilience path):** low-confidence cases can be enhanced when fallback is enabled.

## Phase 7 — Performance, Device Control, and Reliability
- [ ] Add explicit device selection config and auto-fallback CPU behavior.
- [ ] Validate GPU memory budget target (`<= 10 GB`) using selected models.
- [ ] Add stage timing and metadata reporting.

**Depends on:** Phases 3–6  
**Milestone M7 (Operational readiness):** stable CPU/GPU execution with full processing metadata.

## Phase 8 — Delivery Hardening & Acceptance
- [ ] Complete Docker Compose workflow for local run.
- [ ] Add representative test matrix (clean/rotated/perspective/low-quality per doc type).
- [ ] Finalize README architecture rationale + run instructions.

**Depends on:** all prior phases  
**Milestone M8 (Assignment acceptance):** end-to-end success criteria met in compose deployment.

## 5) Testing Strategy Checklist by Phase

- [ ] **Unit tests**
  - [ ] Normalizers (date/name/ID formats)
  - [ ] Validators (plausibility and patterns)
  - [ ] Extractor rules per schema
- [ ] **Integration tests**
  - [ ] API multipart parsing and error codes
  - [ ] Pipeline stage interactions with mocked models
- [ ] **E2E tests**
  - [ ] One sample per document type, then degraded variants
  - [ ] Validate both mandatory artifacts: annotated aligned image + JSON fields
- [ ] **Non-functional checks**
  - [ ] CPU-only execution path
  - [ ] GPU path under memory budget target
  - [ ] Fallback on/off behavior and threshold gating

## 6) Dependency Matrix (What must exist before what)

- [ ] API final payload shape must be defined before deep pipeline integration.
- [ ] Alignment must be reliable before OCR tuning (OCR quality depends on alignment quality).
- [ ] OCR token schema must be stable before extractor rule implementation.
- [ ] Extractor outputs must be stable before validation/confidence aggregation.
- [ ] Confidence metrics must exist before fallback trigger policy can be finalized.
- [ ] Artifact persistence must be in place before end-to-end acceptance tests.

## 7) Suggested Iteration Cadence

- [ ] **Sprint 1:** Phases 0–2 (service foundation + alignment path).
- [ ] **Sprint 2:** Phases 3–4 (OCR + structured extraction for all doc types).
- [ ] **Sprint 3:** Phases 5–6 (validation/confidence + fallback).
- [ ] **Sprint 4:** Phases 7–8 (performance tuning, Docker hardening, acceptance).

## 8) Definition of Done (MVP)

- [ ] `POST /v1/process-document` is fully functional for bank card, ID card, driver’s license.
- [ ] Responses include aligned annotated image + schema-aligned JSON fields.
- [ ] Partial extraction returns explicit nulls with confidence/validation metadata.
- [ ] CPU and GPU modes are supported; GPU path remains within budget target.
- [ ] Docker Compose run is documented and reproducible.
