# 🚀 hello-rust: Accelerate Hierarchical DataFrame Processing with Rust + PyO3

[![Rust](https://img.shields.io/badge/Rust-stable-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyO3](https://img.shields.io/badge/PyO3-enabled-purple.svg)](https://pyo3.rs/)
[![Polars](https://img.shields.io/badge/Polars-DataFrame-0ea5e9.svg)](https://pola.rs/)
[![Conda](https://img.shields.io/badge/Conda-ready-44A833.svg)](https://docs.conda.io/)

> A practical template for writing performance-critical Rust functions and exposing them to Python with **PyO3**, optimized for **Polars DataFrame** workflows and publishable to **Conda**.

---

## 🎯 Repository Purpose

This repository demonstrates how to:

- Build Rust functions for complex data transformations.
- Expose Rust logic to Python via PyO3.
- Use these functions in Python/Polars pipelines for better performance.
- Package and distribute the project through Conda.

The main goal is to speed up heavy hierarchical processing that can become slow and hard to maintain in pure Python.

---

## 😵‍💫 Pain Point This Repo Solves

A common real-world scenario is processing hierarchical maintenance history in tabular form, such as:

- Aircraft component install/remove records
- Parent-child relationship chains
- Unknown hierarchy depth

For each component event, you may need to determine:

> “Is its parent (or ancestor) currently installed on the aircraft at that point in time?”

Doing this with Python loops over large datasets often leads to:

- High runtime cost
- Complex DFS traversal logic
- Difficult-to-maintain code paths

This repo tackles that by moving DFS-style traversal and matching logic into Rust.

---

## 🧠 Core Approach

- **DataFrame layer:** Polars (Python side)
- **Compute layer:** Rust
- **Bridge layer:** PyO3
- **Distribution:** Conda recipe (`recipe/meta.yaml`)

Typical flow:

1. Load/prepare a Polars DataFrame in Python.
2. Call Rust-powered functions for hierarchical traversal and matching.
3. Receive processed results back as DataFrame-ready outputs.

---

## 📊 Example: Before vs After Hierarchical Processing

### Input (Before)

A simplified event history:

| ts | component | parent | event | transaction date |
|---:|:---------:|:------:|:-----:|:----------------:|
| 1 | A320 | AIRCRAFT | Install | 2026-01-01 |
| 2 | ENG-1 | A320 | Install | 2025-12-31 |
| 3 | FAN-9 | ENG-1 | Install | 2025-12-30 |

Challenge: at `2025-12-30`, `FAN-9` installed to parent `ENG-1`, but parent haven't been installed to `AIRCRAFT`.

### Output (After)

After DFS-like hierarchical evaluation:

| ts | component | parent | event | transaction date | new transaction date |
|---:|:---------:|:------:|:-----:|:----------------:|:--------------------:|
| 1 | A320 | AIRCRAFT | Install | 2026-01-01 | 2026-01-01 |
| 2 | ENG-1 | A320 | Install | 2025-12-31 | 2026-01-01 |
| 3 | FAN-9 | ENG-1 | Install | 2025-12-30 | 2026-01-01 |

Explain: All child components should point to the date when it's top parent be installed to `AIRCRAFT`.
This extra result column is what downstream analytics need, but computing it efficiently is where Rust helps.

---

## ✅ Why Rust + PyO3 Here?

- Faster traversal for deep/irregular hierarchies
- Better control over memory and algorithmic behavior
- Reusable from Python with minimal API friction
- Easier scaling to larger maintenance/event datasets

---

## 📦 Conda-Friendly Packaging

This repo includes a Conda recipe so the extension can be distributed and installed in data-science environments.

- Build from Rust + Python packaging metadata
- Ship as Conda artifact for team-wide usage

---

## 🛠️ Tech Stack

- Rust
- PyO3
- Python
- Polars
- Conda

---

## 🧭 Project Direction

Potential next steps:

- Add benchmark comparisons (pure Python vs Rust extension)
- Add end-to-end Polars examples in `examples/`
- Add test fixtures for deep hierarchy and edge cases
- Publish package to internal/public Conda channels

---

## 👀 Who This Is For

- Data engineers dealing with hierarchical tabular data
- Aviation analytics teams (Tracking component usage)
- Python users who need selective Rust acceleration
