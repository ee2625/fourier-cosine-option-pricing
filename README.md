# fourier-cosine-option-pricing

Implementation of the Fang–Oosterlee COS method for European option pricing.

## Project Objective

This project implements the Fourier-Cosine series expansion method introduced in:

**Fang, F. and Oosterlee, C.W.**  
*A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions*

The goal is to correctly implement the COS pricing method for European options, validate the implementation against benchmark models, and study its computational efficiency in terms of accuracy and execution speed.

## Course Context

This repository is a course project on efficient numerical methods in finance.  
The focus is on numerical implementation, validation, robustness testing, and computational efficiency.

## Planned Components

- Implementation of the COS pricing method
- Black-Scholes benchmark pricing
- Characteristic function based pricing setup
- Validation against known benchmark prices
- Robustness testing across model parameters
- Accuracy and runtime comparisons

## Repository Structure

```text
fourier-cosine-option-pricing/
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
│   └── cos_pricing/
│       ├── __init__.py
│       ├── cos_method.py
│       ├── models.py
│       └── utils.py
├── tests/
│   └── test_cos_method.py
├── examples/
│   └── example_european_option.py
└── docs/
    └── paper_notes.md
