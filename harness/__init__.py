"""Resume-Fit harness — Module 1: JD → schema-valid jd_profile.yaml + jd_meta.yaml.

A depth-1 Recursive Language Model (RLM, arXiv 2512.24601): a deterministic root
slices the JD and dispatches focused per-field leaf extractions, then assembles +
validates (via the engine's own redrob_ranker.profile.load) + repairs only the
failing field. Makes a small model reliable at structured output.
"""
