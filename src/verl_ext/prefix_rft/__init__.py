"""Prefix-RFT: blending demonstration and exploration in the RL pipeline.

Implements arXiv:2507.01679v3 on top of the existing GRPO pipeline. Heavy
imports (torch, verl) are deferred to the modules that need them so that the
schedule and the demonstration store can be exercised without the training
stack installed.

Design: docs/superpowers/specs/2026-08-17-prefix-rft-design.md
"""
