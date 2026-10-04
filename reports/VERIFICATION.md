# Verification scope

Checked locally on 4 October 2026 using Python 3.12.14 on Windows:

- **11 tests passed**, including a finite-difference gradient check, extreme-logit behaviour, perceptron stopping, unseen-category transformation and fitted training-set size.
- **11 notebooks completed** from fresh kernels with execution errors disallowed.
- The final notebooks contain **22 static PNG plot outputs**, plus the 3D Plotly display and tabular/text results. No warning or error outputs were retained in the final run.
- The main logistic, classification-tree and regression-tree figures were visually reviewed.
- Dataset bytes were preserved and their SHA-256 recorded in `DATA_SOURCES.md`.

The Jupyter runtime printed local kernel transport warnings to the runner's terminal; those are not notebook execution failures or saved notebook output. Kernels were cleaned up after execution.

Dependency pins describe the tested direct packages; they are not a complete cross-platform dependency lock. The GitHub Actions workflow is provided but has **not** run in the private GitHub repository. The remote Linux run must be checked after upload.

The supplied ZIP has no Git history, remote settings, issues or other branches. A limited common-pattern scan of the prepared files cannot certify that those uninspected surfaces contain no credentials or restricted material. Original course/tutorial attribution remains a maintainer task.
