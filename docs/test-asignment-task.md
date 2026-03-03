# Test Assignment: ML Engineer
Design and implement a system for the initial processing of personal identification documents. The system should align the image, recognize text in the image, and extract structured information from the recognized text (full name, date of birth, etc.)

## What the system must be able to do
    • Accepts an image as input
    • Aligns the image
    • Recognizes text in the image
    • Returns the aligned image with detection results and a JSON with extracted fields

## Technical Requirements
    • Language: Python
    • Models: any open-source model for local inference and any external APIs you see fit
    • Frameworks: PyTorch, OpenCV, any clients for cloud-based models
    • Interface: REST API, you can vibe-code a UI, but it’s not required
    • Build: docker compose

## Constraints
    • We assume the input will be documents in the format of bank cards, ID cards, and driver’s licenses
    • Locally running models must work on both GPU and CPU. GPU memory limit — 10 GB
    • If you use external LLM APIs, you must provide a way to configure the client proxy (base_url) via environment variables.

## Using LLMs During Development
We recommend using LLM tools (ChatGPT, Claude, Copilot, etc.) during development — it’s part of an ML engineer’s workflow. Including request logs or a description of your agent methodology will be a bonus.

## Result
A GitHub repository containing everything needed to run your solution, along with a README file describing the architecture and the rationale behind decisions made (which models were chosen and why). We do not expect a perfect solution, but the core recognition pipeline must be functional. For reference — the solution will be tested on Ubuntu Server 22.04, Docker Compose version v2.25.0, NVIDIA GeForce RTX 2080 Ti, Nvidia Driver 535.171.04