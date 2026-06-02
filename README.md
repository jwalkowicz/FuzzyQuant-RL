# FuzzyQuant-RL

A reinforcement learning framework for algorithmic trading that combines Q-Learning with fuzzy logic state representation. This project uses technical indicators (RSI, MACD) processed through a fuzzy logic system to help the agent make more nuanced trading decisions.

> **Note on Fuzzy Logic Implementation:** The core fuzzy membership functions and rule evaluation are currently **#TBD**. A mock converter is currently used for end-to-end testing.

## Quick Start

This project uses `uv` for lightning-fast dependency management.

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Run the training and evaluation pipeline:**
   ```bash
   uv run python -m src.main
   ```

## Project Structure

- `src/agents/`: Q-Learning agent implementation.
- `src/envs/`: Gymnasium trading environment.
- `src/fuzzy/`: Fuzzy logic state converters (#TBD).
- `src/data/`: Technical indicator processing (using `yfinance` and `ta`).
- `config/`: Environment-based configuration (YAML).

## How it Works

The agent learns to trade by interacting with a simulated market environment. It chooses between three actions: **Hold**, **Buy**, or **Sell**. 
- **Observations:** Discrete states derived from fuzzy-encoded RSI and MACD values.
- **Reward:** The change in total portfolio net worth between steps.
- **Evaluation:** The system automatically compares the trained agent against a simple Buy & Hold baseline on 2025 out-of-sample data.

## Requirements

- Python 3.12+
- `uv` (recommended)
- See `pyproject.toml` for the full list of dependencies.
