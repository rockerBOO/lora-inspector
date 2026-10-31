# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup
```bash
# Use uv to run scripts directly
uv run python lora-inspector.py [options]

# For development and running specific scripts
uv run python filename.py
```

### Code Formatting
```bash
# Format code using black
black .
```

## Project Architecture

### Core Purpose
LoRA Inspector is a tool for analyzing LoRA (Low-Rank Adaptation) machine learning model files, specifically focusing on:
- Extracting metadata from LoRA files
- Analyzing training parameters
- Providing insights into model training characteristics

## Development Notes

### Dependency Management

#### PEP 723 Single-File Dependency Management

For single-file scripts, use the following PEP 723 format:

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "requests>=2.25.1",
#     "pandas",
#     "matplotlib"
# ]
# ///
```

### Script Development Best Practices

#### Argument Parsing
- Always use `argparse` for command-line argument handling
- Define clear, descriptive help text for each argument
- Use appropriate argument types and actions
- Support both required and optional arguments
