# High-Level Vision and Specification

## 1) Product Vision
Build a robust initial document-processing service that accepts an image of a personal identification document, normalizes it for reading, recognizes visible text, and returns structured fields for downstream systems. The service targets practical reliability over perfect extraction and is designed as a local-first inference pipeline with an optional external LLM fallback for difficult cases.

## 2) Goals and Success Criteria
### Primary goals
- Accept one document image per request.
- Align and normalize the document image to improve OCR quality.
- Detect and recognize text on the aligned image.
- Return:
  - an aligned image with detection overlays,
  - structured JSON with extracted fields and confidence metadata.

### MVP success criteria
- End-to-end pipeline runs successfully for representative bank card, ID card, and driver’s license samples.
- Service is exposed through a REST API and runnable via Docker Compose.
- Local inference path runs on CPU and GPU, with GPU usage staying within a 10 GB budget.
- External LLM fallback (if enabled) supports configurable proxy/base URL via environment variables.

## 3) Scope
### In scope
- Document types:
  - Bank cards
  - ID cards
  - Driver’s licenses
- Single-image request processing.
- Field extraction from OCR text into normalized JSON.
- Confidence and validation flags in response.

## 4) System Architecture (High Level)
The service follows a staged pipeline:

1. **Endpoint/API Layer**
	- Receives image upload and request metadata.
	- Performs basic input validation (file type, size, readability).

2. **Preprocessing & Alignment Layer**
	- Detects document boundaries.
	- Applies perspective correction, rotation normalization, and optional denoising/contrast normalization.
	- Produces a canonical aligned image.

3. **Text Detection + OCR Layer**
	- Detects text regions and recognizes text content with local open-source OCR libraries/models.
	- Produces tokens/lines with bounding boxes and confidence values.

4. **Field Extraction Layer**
	- Normalizes OCR text (case, separators, date formats, transliteration rules if configured).
	- Maps recognized text into document-specific schemas.
	- Emits field values and per-field confidence.

5. **Validation & Fallback Layer**
	- Applies consistency checks (date plausibility, known patterns, checksum-style heuristics where applicable).
	- If confidence is below threshold and fallback is enabled, uses external LLM/API extraction path.

6. **Response Composer**
	- Returns structured JSON and aligned annotated image artifact (or encoded payload reference).

## 5) Processing Contract
For each request, processing must follow this order:

1. Validate input image.
2. Align document image.
3. Run OCR on aligned image.
4. Extract structured fields.
5. Validate extracted fields.
6. Optionally invoke fallback extractor.
7. Return final artifacts and metadata.

If alignment fails, return a typed error with failure reason and diagnostic metadata. If OCR succeeds but extraction is partial, return partial fields with explicit nulls and confidence flags rather than hard-failing.

## 6) API Specification (High Level)
### Endpoint
- `POST /v1/process-document`

### Request
- Content type: `multipart/form-data`.
- Required part: `image`.
- Optional parameters:
  - `document_type_hint` (`bank_card` | `id_card` | `drivers_license` | `auto`)
  - `use_external_fallback` (boolean)

### Response (success)
- `request_id`
- `document_type_detected`
- `aligned_image` (path, URL, or base64 payload depending on deployment mode)
- `detections` (text boxes/polygons + confidence + recognized text)
- `fields` (structured values by schema)
- `field_confidence`
- `validation_flags`
- `processing_metadata` (latency, model versions, device used: CPU/GPU, fallback usage)

### Error model
- `400` invalid input/unsupported media.
- `422` unreadable or non-document image.
- `500` internal processing failure.
- All error responses include machine-readable `error_code` and human-readable `message`.

## 7) Output Requirements
The system returns two mandatory artifacts:

1. **Aligned image with detection results**
	- Corrected document orientation/perspective.
	- Corrected document orientation/perspective with visual overlays for detected text regions for debugging purposes.

2. **JSON with extracted fields**
	- Schema-aligned values.
	- Confidence and validation metadata.

## 8) Document Schemas
### 8.1 Bank Card Schema
Required fields:
- `card_number` (masked/unmasked policy configurable)
- `cardholder_name`
- `expiry_date`

Optional fields:
- `issuer_network` (Visa/Mastercard/МИР/etc.)
- `bank_name`

### 8.2 ID Card Schema
Required fields:
- `full_name`
- `date_of_birth`
- `sex`
- `place_of_birth`
- `document_number`

Optional fields:
- `issuing_authority`
- `issue_date`
- `expiry_date` (if present on document class)

### 8.3 Driver’s License Schema
Required fields:
- `full_name`
- `date_of_birth`
- `place_of_birth`
- `issue_date`
- `expiry_date`
- `issuing_authority`
- `license_number`
- `place_of_residence`
- `license_class`


### Normalization rules (all schemas)
- Dates normalized to ISO-like format where possible (`YYYY-MM-DD`), otherwise returned with raw value + parse status.
- Names returned in normalized spacing/casing plus raw OCR variant.
- Nulls are explicit when fields are missing/unreadable.

## 9) Model and Runtime Strategy
### Local inference (primary)
- Use open-source OCR libraries/models for alignment support (OpenCV-based geometry) and OCR.
- Must run on both CPU and GPU environments.

### GPU/CPU compatibility constraints
- GPU memory budget target: <= 10 GB.
- Provide device selection by configuration.
- Provide safe CPU fallback when GPU is unavailable.

### External API/LLM (optional fallback)
- Used only when local confidence is below configured thresholds.
- Client settings must be configurable via environment variables, including base URL/proxy endpoint.

## 10) Deployment and Configuration
### Build and runtime
- Delivered as a Python service with Docker Compose.
- Compatible with Ubuntu Server 22.04 runtime expectations.

### Key configuration groups
- Model/device settings (CPU/GPU selection).
- Confidence thresholds for extraction/fallback.
- External API settings including `base_url`.
- Logging and observability controls.

## 11) Quality, Evaluation, and Acceptance
### Functional validation
- Test each document type with clean, rotated, perspective-distorted, and low-quality images.
- Verify output includes both aligned annotated image and JSON fields.

### Suggested quality metrics
- Alignment success rate.
- OCR character/word accuracy (or proxy metric).
- Field-level precision/recall for required fields.
- End-to-end success rate (all required fields extracted or flagged correctly).

### Acceptance for assignment
- Core pipeline works end-to-end in Docker Compose.
- REST API returns required artifacts.
- README explains architecture and rationale for model choices.

## 12) Risks and Mitigations
- **Layout diversity risk:** Card templates vary by issuer/country.
  - Mitigation: document-type hinting + flexible extraction rules.
- **OCR degradation risk:** blur/glare/compression artifacts reduce text quality.
  - Mitigation: stronger preprocessing and confidence-aware fallback.
- **Partial field ambiguity:** names/dates can be noisy or conflicting.
  - Mitigation: validation flags, raw text preservation, and explicit uncertainty.

## 13) Deliverables
- Source repository with runnable service.
- Docker Compose setup.
- README with architecture and model-selection rationale.
- This specification as the high-level product and technical contract.