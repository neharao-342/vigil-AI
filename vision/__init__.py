"""L3 — Real vision backends (kept separate from the dependency-light
`engines/` package so the graph stays importable without ultralytics/cv2
installed). Everything here is loaded lazily and injected into blocks via
config['detector'], never imported by `engines/blocks.py` directly.
"""
