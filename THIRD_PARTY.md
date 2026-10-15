# Third-party notices

BehaviorLab application code uses the repository's MIT license.

* ByteTrack: Copyright (c) 2021 Yifu Zhang, MIT. Vendored `byte_tracker.py`, `basetrack.py`,
  `kalman_filter.py` from https://github.com/ifzhang/ByteTrack. Adapted imports, NumPy dtype,
  new-track threshold, removed-track cleanup and NumPy linear algebra. `matching.py` provides
  a NumPy assignment and IoU implementation. Source license: `behaviorlab/vendor/ByteTrack-LICENSE.txt`;
  the desktop distribution also provides `licenses/ByteTrack-LICENSE.txt`.
* Python, Tcl/Tk, NumPy, OpenCV and Pillow: distributed runtime components; license notices
  are copied into `licenses/` and retained in `_internal` where provided by their packages.
* Model: converted from the user's `best.pt`, published at
  https://github.com/dachengac/locust-oviposition/releases/tag/v1.0 . Source SHA-256:
  `56e3f750492ff16927142ff01e1545a82b073e1e4435c73e824c0c9475b78de8`.
  Export performed with Ultralytics YOLO (https://github.com/ultralytics/ultralytics,
  AGPL-3.0 / commercial license). Model provenance and upstream license information are
  supplied as `PROVENANCE.txt`, `SOURCE-LICENSE.txt` and `EXPORTER-LICENSE.txt` in the default model pack.
  The exporter notice identifies the tool used and its upstream terms; the application's MIT license
  does not replace third-party terms. Training and export dependencies are not included in the runtime.

BORIS and SimBA documentation informed the workflow. Their application code is not included.
