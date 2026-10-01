# V4 result-manifest hash reconciliation

Classification: **A — byte-formatting-only difference**.

| Representation | SHA-256 |
|---|---|
| Committed Git blob (archival bytes, LF) | `0591d54f1e5c45fbb7c5a1910b6cd01a045424848a81ae33d9b477f021ed522e` |
| Prior Windows working-tree bytes (CRLF) | `58c2ece8410574de388b0933e17a0b8295e65834f74858edf7d21956e9ce3ff8` |

The manifest entered Git in commit `9e979ae323190117c94723eb3f44a7a3e850fa29` and has no later content-changing commit. The Git blob is 6,156 bytes with 103 LF line endings. A Windows checkout under system `core.autocrlf=true` expanded those lines to CRLF, producing 6,259 working-tree bytes and the second hash. Replacing each CRLF with LF reproduces the archival hash exactly; parsed JSON and all scientific/result fields are identical.

The archival hash is `0591d54f…`, matching the paper and freeze test. `.gitattributes` now marks this one frozen manifest `-text`, so Git checks it out byte-for-byte on every platform. No expected hash, scientific result, provenance field, or historical file was rewritten.
