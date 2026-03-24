# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project uses conventional
commit messages for milestone commits.

## [Unreleased]

### Added

- Initial project charter and milestone plan
- Repository standards, milestone ledger, changelog discipline, and developer operating instructions
- Packaging, linting, typing, testing, and pre-commit gate configuration
- A local git-hook installer and a repository contract test
- The initial `kyma` package layout with explicit data, model, training, evaluation, and CLI module boundaries
- Typed model and evaluation protocol configuration surfaces with packaged JSON defaults
- Basic CLI utilities for listing and printing packaged configurations
- Architecture documentation and configuration-focused tests
- Piece-level data adapters for Aria-compatible tokenization, canonical encoded piece records, and tempo-map-aware token timing features
- Focused tests for tempo maps, timing-feature extraction, and MIDI-to-piece tokenization

### Changed

- Expanded the README to reflect the actual project thesis
- Scoped data-directory ignore rules to repo-root artifacts so package modules remain trackable
