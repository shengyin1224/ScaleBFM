# HAT4 FOCUS v2

This is the 3000-episode HAT4 corpus with embodiment-aware FOCUS labels.
BrainCo uses physical closure channels `[0,2,3,4,5]` per hand scaled by the 0.8 hardware close value; Inspire uses `1-state` on channels `[0,1,2,3,4]` per hand. Thumb side/yaw controls are excluded, channels must cross the hysteresis band, hands are processed independently, and episode boundaries are censored. Body arrays are bit-identical to the legacy HAT4 package.

Training manifests are under `manifests/` (2400 train, 600 heldout, 3000 all).
Two frame-aligned HDF5 validation renders and their exact FOCUS intervals are
under `validation_videos/`.
