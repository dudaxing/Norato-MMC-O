# License and attribution

This reproduction is distributed under the
[Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

It is based on:

- Hollis Smith and Julian A. Norato, *A MATLAB code for topology
  optimization using the geometry projection method*, Structural and
  Multidisciplinary Optimization 62 (2020), 1579-1594,
  https://doi.org/10.1007/s00158-020-02552-0.
- The authors' MATLAB educational implementation,
  https://github.com/jnorato/GPTO.
- Andres Ortegon's Python migration,
  https://github.com/jnorato/PyGPTO.

No source file from PyGPTO's GPL-licensed `MMA.py` is copied into this
repository.  The single-constraint MMA routine in `src/gpto/optimizer.py`
is an independent implementation of the separable reciprocal subproblem
needed by the three examples.  All newly written reproduction, testing,
command-line, and reporting code is offered under CC BY-NC 4.0.
