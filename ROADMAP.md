# FMS Project Roadmap

This document outlines the future direction of the Foundation Model Stack (FMS) project. It is a living document and will be updated as the project evolves.

## Model Support

Our goal is to expand the range of models supported by FMS, and to provide full support for all models.

- **Complete Support for Existing Models:**
  - [ ] Add training and tuning support for GPT-BigCode.
  - [ ] Add training and tuning support for RoBERTa.

- **New Model Architectures:**
  - [ ] Add support for Mixtral.
  - [ ] Add support for Gemma.

## Core Infrastructure

We are continuously working to improve the core infrastructure of FMS.

- **Performance and Stability:**
  - [ ] Work with the PyTorch team to resolve the open issue preventing `torch.compile` from working with training/finetuning.
  - [ ] Improve inference stability and reduce memory footprint.

- **Quantization:**
  - [ ] Expand support for different quantization methods (e.g., AWQ, GPTQ).

## Community and Documentation

We want to make FMS more accessible and easier to use for the community.

- **Documentation:**
  - [ ] Create more comprehensive tutorials and examples for common use cases.
  - [ ] Improve the API documentation to make it easier to understand and use.
