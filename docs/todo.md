# TODO — Dependency-Ordered Implementation Backlog

This backlog is decomposed from `docs/implementation-plan.md` into small,
isolated, verifiable tasks.

## Execution Rules

- Implement Test-Driven Development (TDD) principles. Write tests first, 
  verify they fail, then generate the code to pass those tests.
- Complete tasks in ID order unless dependency notes allow parallel work.
- Every task is done only when its `Verify` condition passes.
- Keep each PR/task scoped to one item when possible.
- Use the workspace-configured Python environment.

## Task Format

- **Depends on:** Required task IDs that must be complete first.
- **Implement:** Small, concrete code/config change.
- **Verify:** Independent test or check for this task.

---

## Phase 0 — Foundation & Scaffolding

- [x] **T001** Initialize Python service skeleton and package layout.
  - Depends on: None
  - Implement: Create `src/app/*` and `tests/*` folders from the approved
    structure.
  - Verify: `pytest --collect-only` runs without import/path errors.

- [x] **T002** Add dependency/project management and basic tooling config.
  - Depends on: T001
  - Implement: Add project/dependency files and test runner configuration.
  - Verify: Environment installs; `pytest --collect-only` succeeds.

- [x] **T003** Implement environment config model.
  - Depends on: T002
  - Implement: Add `src/app/core/config.py` with typed settings for device,
    fallback, artifact paths, and thresholds.
  - Verify: Unit tests for default values and env overrides pass.

- [x] **T004** Add structured logging bootstrap.
  - Depends on: T003
  - Implement: Add `src/app/core/logging.py` and configure application logger.
  - Verify: Unit test confirms expected structured fields are emitted.

- [x] **T005** Add core exception primitives.
  - Depends on: T003
  - Implement: Add typed exceptions in `src/app/core/exceptions.py`.
  - Verify: Unit tests confirm exception categories and payload fields.

- [x] **T006** Add container runtime baseline.
  - Depends on: T002
  - Implement: Add `Dockerfile`, `compose.yaml`, and `.env.example`.
  - Verify: `docker compose config` validates successfully.

- [x] **T007** Add app bootstrap and health route.
  - Depends on: T004, T005
  - Implement: Wire app entrypoint in `src/app/main.py` and health endpoint.
  - Verify: Integration test for health endpoint returns 200.

---

## Phase 1 — API Contract & Error Model

- [x] **T008** Add request ID generation and propagation.
  - Depends on: T007
  - Implement: Add middleware that injects `request_id` to context/response.
  - Verify: Integration test confirms request ID exists and is stable per request.

- [x] **T009** Implement request schema contract.
  - Depends on: T007
  - Implement: Add multipart contract schema in `src/app/api/schemas.py` for
    `image`, `document_type_hint`, `use_external_fallback`.
  - Verify: Unit tests for accepted/rejected request payloads pass.

- [x] **T010** Implement success response schema.
  - Depends on: T009
  - Implement: Add `ProcessDocumentResponse` fields including nullable
    extraction fields.
  - Verify: Serialization tests validate required keys and nullable behavior.

- [x] **T011** Implement typed error response models.
  - Depends on: T005
  - Implement: Add API error models for 400/422/500 in `src/app/api/errors.py`.
  - Verify: Unit tests confirm error envelopes include `error_code` and
    `message`.

- [x] **T012** Register global exception handlers.
  - Depends on: T011, T007
  - Implement: Map core exceptions to typed API error responses.
  - Verify: Integration tests confirm status code mapping for representative
    errors.

- [x] **T013** Implement pipeline context and result models.
  - Depends on: T010
  - Implement: Add `PipelineContext` and `PipelineResult` in
    `src/app/pipeline/context.py` and `src/app/pipeline/result.py`.
  - Verify: Unit tests validate required fields, diagnostics, and timing
    containers.

- [x] **T014** Implement pipeline orchestrator skeleton.
  - Depends on: T013
  - Implement: Add `process_document_pipeline` in
    `src/app/pipeline/processing.py` with stage stubs in strict order.
  - Verify: Unit test asserts stage order and short-circuit semantics.

- [x] **T015** Implement `POST /v1/process-document` placeholder route.
  - Depends on: T008, T009, T010, T012, T014
  - Implement: Add endpoint handler in `src/app/api/routes.py` returning
    contract-valid placeholder payload.
  - Verify: Integration test validates response shape and error behavior.

---

## Phase 2 — Input Validation & Alignment

- [x] **T016** Implement image load/decode helpers.
  - Depends on: T014
  - Implement: Add `src/app/preprocessing/image_io.py` for safe byte-to-image
    decoding.
  - Verify: Unit tests for valid decode and decode failure paths.

- [x] **T017** Implement type and size validation.
  - Depends on: T016
  - Implement: Add media type and file-size checks in
    `DocumentPreprocessor.validate_type_size_readability`.
  - Verify: Unit tests cover allowed and blocked content types/sizes.

- [x] **T018** Implement readability checks.
  - Depends on: T016
  - Implement: Add readability heuristics for blank/corrupt/non-document-like
    inputs.
  - Verify: Unit tests classify unreadable fixtures as invalid.

- [x] **T019** Implement document boundary detection.
  - Depends on: T016
  - Implement: Add boundary detection in
    `src/app/preprocessing/document_preprocessor.py`.
  - Verify: Fixture tests detect corners/bounds on representative documents.

- [x] **T020** Implement perspective correction.
  - Depends on: T019
  - Implement: Apply geometric transform to create canonical aligned image.
  - Verify: Fixture tests confirm corrected output dimensions/orientation.

- [x] **T021** Implement rotation normalization.
  - Depends on: T020
  - Implement: Normalize orientation after perspective correction. The angle of rotation of document on image can be arbitrary.
  - Verify: Tests with rotated fixtures yield upright alignment.

- [x] **T022** Implement denoise and contrast normalization.
  - Depends on: T021
  - Implement: Add optional denoise/contrast stage.
  - Verify: Unit tests verify transform output and no-crash behavior.

- [x] **T023** Add typed alignment diagnostics.
  - Depends on: T020
  - Implement: Return structured alignment failure diagnostics.
  - Verify: Unit tests validate diagnostic payload for known failure cases.

- [ ] **T024** Wire preprocessing stage into pipeline.
  - Depends on: T017, T018, T022, T023
  - Implement: Replace preprocess stub in pipeline orchestrator.
  - Verify: Integration test confirms aligned artifact data exists in pipeline
    result.

---

## Phase 3 — OCR Detection & Recognition

- [ ] **T025** Define OCR domain models and detector interface.
  - Depends on: T024
  - Implement: Add `TextRegion` model and `TextDetector` contract.
  - Verify: Unit tests for model validation and detector return typing.

- [ ] **T026** Implement detector adapter.
  - Depends on: T025
  - Implement: Add local OCR detector integration in
    `src/app/ocr/detector.py`.
  - Verify: Fixture smoke test returns text regions with confidence.

- [ ] **T027** Define recognizer models and interface.
  - Depends on: T025
  - Implement: Add `RecognizedToken`, `RecognizedLine`, recognizer contract.
  - Verify: Unit tests validate token/line model constraints.

- [ ] **T028** Implement recognizer adapter.
  - Depends on: T027
  - Implement: Add OCR recognition call in `src/app/ocr/recognizer.py`.
  - Verify: Fixture test returns recognized tokens for detected regions.

- [ ] **T029** Implement token-to-line grouping.
  - Depends on: T028
  - Implement: Add `group_tokens_to_lines` for deterministic line assembly.
  - Verify: Unit tests with synthetic tokens assert expected line grouping.

- [ ] **T030** Wire OCR stage into pipeline.
  - Depends on: T026, T029
  - Implement: Replace OCR stub with detector+recognizer flow.
  - Verify: Integration test confirms detections and OCR lines in pipeline
    output.

---

## Phase 4 — Extraction & Normalization

- [ ] **T031** Add extraction base interface.
  - Depends on: T030
  - Implement: Add `BaseExtractor` and extracted fields model.
  - Verify: Unit tests validate extractor contract and base behavior.

- [ ] **T032** Implement shared extraction rule helpers.
  - Depends on: T031
  - Implement: Add regex/synonym/token cleanup helpers in
    `src/app/extraction/rules_common.py`.
  - Verify: Unit tests for helper correctness and edge cases.

- [ ] **T033** Implement normalization utilities.
  - Depends on: T032
  - Implement: Add `normalize_date`, `normalize_name`, and
    `normalize_document_number`.
  - Verify: Unit tests for parse status, formatting, and fallback behavior.

- [ ] **T034** Implement bank card extractor.
  - Depends on: T033
  - Implement: Add `BankCardExtractor` required/optional mapping logic.
  - Verify: Unit tests with bank card OCR fixtures.

- [ ] **T035** Implement ID card extractor.
  - Depends on: T033
  - Implement: Add `IdCardExtractor` required/optional mapping logic.
  - Verify: Unit tests with ID card OCR fixtures.

- [ ] **T036** Implement driver license extractor.
  - Depends on: T033
  - Implement: Add `DriversLicenseExtractor` required/optional mapping logic.
  - Verify: Unit tests with license OCR fixtures.

- [ ] **T037** Implement document type dispatcher.
  - Depends on: T034, T035, T036
  - Implement: Add hint/auto extractor routing in
    `src/app/extraction/dispatcher.py`.
  - Verify: Unit tests for all hint and auto-detection branches.

- [ ] **T038** Wire extraction stage into pipeline.
  - Depends on: T037
  - Implement: Replace extraction stub with dispatcher+extractor invocation.
  - Verify: Integration tests for all 3 document types produce structured
    fields.

---

## Phase 5 — Validation & Confidence

- [ ] **T039** Implement field-level validators.
  - Depends on: T038
  - Implement: Add date plausibility and pattern checks in
    `src/app/validation/field_validators.py`.
  - Verify: Unit tests for valid/invalid date and number patterns.

- [ ] **T040** Implement cross-field consistency checks.
  - Depends on: T039
  - Implement: Add `ConsistencyChecks` for conflicting/mismatched fields.
  - Verify: Unit tests for consistency flag generation.

- [ ] **T041** Implement confidence aggregation.
  - Depends on: T039
  - Implement: Add `ConfidenceScorer` in `src/app/validation/confidence.py`.
  - Verify: Unit tests for per-field and aggregate confidence calculations.

- [ ] **T042** Wire validation/confidence stage into pipeline.
  - Depends on: T040, T041
  - Implement: Replace validation stub with validators, checks, and scoring.
  - Verify: Integration test confirms `validation_flags` and
    `field_confidence` are populated.

- [ ] **T043** Enforce partial extraction contract with explicit nulls.
  - Depends on: T042, T010
  - Implement: Ensure missing fields remain explicit `null` in response model
    and pipeline result.
  - Verify: Integration test confirms partial extractions return 200 with null
    fields.

---

## Phase 6 — Optional External Fallback

- [ ] **T044** Implement fallback client with env-configurable base URL/proxy.
  - Depends on: T003
  - Implement: Add `src/app/fallback/client.py` with configurable endpoint and
    credentials.
  - Verify: Unit tests validate client initialization from env and request
    shaping.

- [ ] **T045** Implement fallback prompt builder.
  - Depends on: T038, T044
  - Implement: Add `src/app/fallback/prompt_builder.py` with stable structured
    prompt payload.
  - Verify: Unit tests confirm required context is present in payload.

- [ ] **T046** Implement fallback adapter merge logic.
  - Depends on: T045, T043
  - Implement: Add `src/app/fallback/adapter.py` to merge fallback output with
    provenance flags.
  - Verify: Unit tests for merge precedence and provenance output.

- [ ] **T047** Implement fallback invocation policy.
  - Depends on: T041, T046
  - Implement: Add threshold-based policy in `src/app/fallback/policy.py`.
  - Verify: Unit tests for invoke/skip cases across threshold boundaries.

- [ ] **T048** Wire fallback stage into pipeline.
  - Depends on: T047
  - Implement: Integrate optional fallback call in orchestrator when enabled and
    low confidence.
  - Verify: Integration tests for fallback off/on and low-confidence triggering.

---

## Phase 7 — Device Control, Performance, Reliability

- [ ] **T049** Implement artifact storage manager.
  - Depends on: T024
  - Implement: Add request-scoped artifact directory creation and persistence in
    `src/app/storage/artifacts.py`.
  - Verify: Integration test verifies aligned image path persistence.

- [ ] **T050** Implement detection overlay rendering.
  - Depends on: T030, T049
  - Implement: Add `draw_detections` and persist annotated image.
  - Verify: Integration test confirms overlay artifact exists and is readable.

- [ ] **T051** Implement per-stage latency metrics.
  - Depends on: T014
  - Implement: Add `MetricsCollector` in `src/app/telemetry/metrics.py`.
  - Verify: Unit tests validate stage timing accumulation and totals.

- [ ] **T052** Implement trace context metadata.
  - Depends on: T051, T048
  - Implement: Add `TraceContext` in `src/app/telemetry/tracing.py` for device,
    model versions, fallback usage.
  - Verify: Integration test confirms processing metadata fields are present.

- [ ] **T053** Add device selector (`cpu`, `cuda`, `auto`) and safe CPU
  fallback.
  - Depends on: T026, T028, T003
  - Implement: Add runtime device routing in config/model-loading paths.
  - Verify: Unit tests with mocked availability validate selection/fallback.

- [ ] **T054** Add GPU memory budget guardrail (`<= 10 GB`).
  - Depends on: T053
  - Implement: Add model/config guard checks and fail-fast diagnostics for
    over-budget runtime.
  - Verify: Non-functional tests validate reject/allow behavior for configured
    budgets.

---

## Phase 8 — Delivery Hardening & Acceptance

- [ ] **T055** Finalize API response composition with required artifacts.
  - Depends on: T050, T052, T043
  - Implement: Ensure response returns aligned artifact reference, detections,
    fields, confidence, flags, and processing metadata.
  - Verify: Integration test validates full contract for success and partial
    cases.

- [ ] **T056** Add integration tests for multipart parsing and error codes.
  - Depends on: T015, T055
  - Implement: Add tests for 400/422/500 and valid request paths.
  - Verify: Integration test suite passes.

- [ ] **T057** Add unit suites for normalizers, validators, extractors.
  - Depends on: T033, T039, T037
  - Implement: Expand unit tests for core deterministic logic.
  - Verify: Unit test suite passes with required coverage targets.

- [ ] **T058** Add E2E matrix by document type and quality condition.
  - Depends on: T055
  - Implement: Add E2E cases for clean/rotated/perspective/low-quality
    documents for bank card, ID card, and driver license.
  - Verify: E2E suite passes and asserts both mandatory artifacts.

- [ ] **T059** Finalize Docker Compose workflow for local run.
  - Depends on: T055, T058
  - Implement: Harden compose service definitions, volumes, env wiring, and
    startup checks.
  - Verify: `docker compose up` produces a working API service end-to-end.

- [ ] **T060** Finalize README architecture rationale and run instructions.
  - Depends on: T059
  - Implement: Document architecture, model choices, device mode behavior,
    fallback config, and test/run steps.
  - Verify: Fresh setup from README succeeds on a clean machine.

---

## Milestone Gates (Cross-Check)

- [ ] **M0** Ready-to-code runtime: T001–T007
- [ ] **M1** Contract-first API: T008–T015
- [ ] **M2** Alignment stage complete: T016–T024
- [ ] **M3** OCR stage complete: T025–T030
- [ ] **M4** Structured extraction complete: T031–T038
- [ ] **M5** Trustworthy outputs: T039–T043
- [ ] **M6** Resilience path: T044–T048
- [ ] **M7** Operational readiness: T049–T054
- [ ] **M8** Assignment acceptance: T055–T060