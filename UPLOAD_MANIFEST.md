# VTF-Flow public-upload manifest

This directory is the source-code package intended for the public GitHub
repository. It was copied from the research workspace without raw data,
generated outputs, checkpoints, prediction archives, Python bytecode, or local
tool metadata.

## Included

- all runtime Python modules;
- frozen JSON experiment configurations;
- preprocessing, training, evaluation, and plotting scripts;
- unit and synthetic numerical tests;
- data/coordinate/method documentation;
- reference main-table and field-diagnostic values;
- Python packaging and dependency metadata;
- a GitHub Actions unit-test workflow.

## Intentionally excluded

- the third-party RELLIS-3D dataset;
- `data/` and `outputs/`;
- `.npy`, `.npz`, and checkpoint artifact bundles;
- generated manuscript figures and temporary debug results;
- local Codex/Nature tooling metadata and caches.

## Required author actions before making the repository public

1. Add a software `LICENSE` approved by the authors/institutions.
2. Add `CITATION.cff` after the paper title, repository URL, and author order
   are final.
3. Publish the cache, frozen models/predictions, and reference-result bundle in
   a versioned GitHub Release or archival repository.
4. Add the artifact URL, DOI (when available), version, and SHA-256 checksums to
   `README.md`.
5. Run the README workflow once on a clean machine before submission.

The absence of a software licence is deliberate at this stage: no permission
terms were inferred on behalf of the authors.
