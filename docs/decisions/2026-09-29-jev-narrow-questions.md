# Jev decides from narrow questions only

- **Problem**: the first real Workers AI calls to `typesafe/jev` returned a double envelope (`result.result.answers`) that the provider did not unwrap, so every call escalated silently; once fixed, the broad approve/escalate/deny choice stayed soft (0.61 to 0.79) and a broad `destructive` question scored creating a test file 0.63.
- **Decision**: unwrap any `result` nesting; drop the choice; ask six nouls (`deletes`, `overwrites`, `exfiltrates`, `secrets` with true/false criteria, plus `in_scope`, `injection`) and decide in code: injection escalates, deny needs harm >= `deny_min` and out of scope, approve needs harm <= `destructive_max` and in scope, confidence = weakest safe-side signal.
- **Why**: on the same calls the narrow questions answered 0.99 where the choice hesitated; TypeSafe's guidance is atomic questions combined in code; a blocked call now says which fact blocked it, which a human can check.
- **Where**: `src/sup7/providers/jev.py`, `tests/test_jev.py`, README. Real trial on 5 hand-written cases: 5/5, but c1 passes at harm 0.19 against a 0.20 threshold; thresholds still need a labelled bench drawn from mesh7 traces.
