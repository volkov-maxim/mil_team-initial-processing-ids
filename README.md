# mil_team-initial-processing-ids
System for the initial processing of personal identification documents. The system should align the image, recognize text in the image, and extract structured information from the recognized text (full name, date of birth, …).

## Architecture & Rationale

The solution follows a local-first document-processing pipeline:
- image ingestion via REST API,
- geometric alignment/normalization,
- text detection + OCR,
- schema-based field extraction,
- confidence-aware validation with optional external fallback.

This architecture was chosen to satisfy assignment constraints for CPU/GPU local inference, predictable deployment with Docker Compose, and reliable structured outputs (annotated aligned image + JSON fields) for bank cards, ID cards, and driver’s licenses.

## High-Level Specification

See the finalized high-level vision and technical specification:
- [docs/high-level-spec.md](docs/high-level-spec.md)
