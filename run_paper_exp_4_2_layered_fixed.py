"""Stable command-line entry point for the corrected layered rolling test."""

# The v2 module installs the corrected executor and invokes the base parser
# during import.  Do not call a second ``main`` function after importing it.
import run_layered_paper_exp_v2  # noqa: F401
