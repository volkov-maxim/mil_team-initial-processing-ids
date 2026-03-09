# mil_team-initial-processing-ids

Local-first REST service for initial processing of personal identification
documents. The service:
- accepts one image per request,
- aligns and normalizes the document,
- runs OCR,
- extracts schema-aligned fields,
- returns both JSON fields and artifact paths.

Supported document families:
- `bank_card`
- `id_card` (Russian passport-style)
- `drivers_license` (Russian driver's license)

## Architecture and Rationale

The pipeline follows the contract from `docs/high-level-spec.md`:
1. Validate input image.
2. Align document image.
3. Run OCR on aligned image.
4. Extract fields by document schema.
5. Validate fields and compute confidence.
6. Optional external fallback stage.
7. Compose API response.

Why this design:
- `FastAPI` provides a typed API contract and stable error handling.
- `OpenCV` is used for deterministic geometric alignment and preprocessing.
- `EasyOCR` + `torch` provide local OCR with CPU/GPU execution.
- `Pydantic` models keep request/response and pipeline contracts strict.

## Runtime Behavior

### Device mode and GPU guardrail

Configuration keys:
- `APP_DEVICE_MODE`: `cpu` | `cuda` | `auto`
- `APP_GPU_MEMORY_BUDGET_GB`: hard-capped to `<= 10.0`

Behavior:
- `cpu`: always run OCR on CPU.
- `cuda`: use GPU when available, otherwise safe CPU fallback.
- `auto`: prefer GPU if available, else CPU.
- If configured GPU budget is invalid (`> 10 GB` or below model requirement),
  the service fails fast with a typed internal error.

### Fallback configuration

Environment variables are available for optional external fallback wiring:
- `APP_USE_EXTERNAL_FALLBACK_DEFAULT`
- `APP_FALLBACK_BASE_URL`
- `APP_FALLBACK_PROXY_URL`
- `APP_FALLBACK_CONFIDENCE_THRESHOLD`
- `APP_REQUIRED_FIELD_CONFIDENCE_THRESHOLD`

Current status: fallback telemetry/config fields are present; fallback execution
logic is scaffolded and currently a no-op stage in the pipeline.

## Fresh Setup (Local)

Prerequisites:
- Python `3.12`
- `pip`

1. Create and activate a virtual environment.

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows (Git Bash)
source .venv/Scripts/activate
```

PowerShell alternative:

```powershell
.venv\Scripts\Activate.ps1
```

2. Install project dependencies.

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,ocr]"
```

3. Create local environment file.

```bash
cp .env.example .env

# Windows (Command Prompt)
copy .env.example .env
```

4. Run the API.

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

5. Verify health.

```bash
curl http://127.0.0.1:8000/health
```

## Fresh Setup (Docker Compose)

1. Create `.env` from the example.

```bash
cp .env.example .env
```

2. Start CPU/default service.

```bash
docker compose up --build
```

3. Optional GPU profile.

```bash
docker compose --profile gpu up --build api-gpu
```

## API Usage

Endpoint:
- `POST /v1/process-document` (`multipart/form-data`)

Required part:
- `image`

Optional form fields:
- `document_type_hint`: `bank_card` | `id_card` | `drivers_license` | `auto`
- `use_external_fallback`: `true` | `false`

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/v1/process-document" \
  -F "image=@images/id_cards/passport_min.png" \
  -F "document_type_hint=auto" \
  -F "use_external_fallback=false"
```

Output includes:
- `aligned_image`
- `detections`
- `fields`
- `field_confidence`
- `validation_flags`
- `processing_metadata`

Artifacts are persisted under `artifacts/<request_id>/`.

## Tests

Pytest is configured in `pyproject.toml` with `--pspec` output formatting.

Run all tests:

```bash
python -m pytest
```

Run by suite:

```bash
python -m pytest tests/unit
python -m pytest tests/integration
python -m pytest tests/e2e
```

## References

- `docs/high-level-spec.md`
- `docs/implementation-plan.md`
- `docs/todo.md`
