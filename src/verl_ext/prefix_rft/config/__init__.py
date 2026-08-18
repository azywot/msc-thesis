"""Hydra config package for Prefix-RFT.

This file is not optional and it is not a formality. ``entrypoint.py`` declares
``@hydra.main(config_path="pkg://verl_ext.prefix_rft/config", ...)``, and Hydra
resolves a ``pkg://`` path to an importable module — so the directory holding
``prefix_rft_trainer.yaml`` has to be a package. Without this file the launcher dies
at startup with:

    Primary config module 'verl_ext.prefix_rft.config' not found.
    Check that it's correct and contains an __init__.py file

The vendored AgentFlow entrypoint has the same arrangement
(``fine_tuning/agentflow/verl/__init__.py``), for the same reason.
"""
