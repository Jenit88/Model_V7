#!/usr/bin/env python3
"""Run model_v7's self-test on CPU, without disturbing any live run."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("model_v7", HERE / "model_v7.py")
module = importlib.util.module_from_spec(spec)
# tta_v7 imports `tta`, and the evaluation path imports `tta_v7` by name, so
# the module has to be registered before it executes.
sys.modules["model_v7"] = module
spec.loader.exec_module(module)

module.REQUIRE_GPU = False
module.run_selftest()
