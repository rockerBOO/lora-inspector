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

### Key Functionalities
- Inspect LoRA file metadata
- Save metadata to JSON
- Analyze weight magnitudes
- Explore tag frequencies
- Update metadata for LoRA files

### Primary Scripts
- `lora-inspector.py`: Main inspection tool
- `update_metadata.py`: Metadata modification utility

### Supported File Types
- Primary support for `.safetensors` files
- Primarily works with Kohya-ss LoRA metadata
- Focuses on Stable Diffusion model training metadata

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

#### Project Dependencies
- Uses `uv` for script and dependency management
- Requires Python 3.11+
- Key dependencies: safetensors, pillow, numpy, torch
- Recommended to use `uv run python filename.py` for running scripts

### Script Development Best Practices

#### Argument Parsing
- Always use `argparse` for command-line argument handling
- Define clear, descriptive help text for each argument
- Use appropriate argument types and actions
- Support both required and optional arguments

### Future Development
Potential areas for expansion:
- Metadata database integration
- Enhanced LoRA file comparison
- Comprehensive training issue detection