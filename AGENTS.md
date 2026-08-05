# AGENTS.md

Project instructions for AI coding agents working in this repository.

## Project Overview

Speech data processing pipeline. This repository is under active development and the initial structure is not yet defined.

## Commands

> TODO: Add lint, test, and build commands once the project structure and language are decided.

## Conventions

- No code exists yet — agree on structure (package manager, directory layout) before writing code.
- Keep data processing steps reproducible: prefer configuration over hardcoded paths.
- Do not commit large audio or data assets; use `.gitignore` entries for data directories and generated artifacts.
- Do not commit API keys, tokens, or other secrets. Never log secrets.

## Guidelines for Agents

- Before making structural decisions, check existing files and follow conventions already in place.
- Verify changes with the project's test/lint commands before finishing a task.
- Ask the user before installing new dependencies or changing the project architecture.
