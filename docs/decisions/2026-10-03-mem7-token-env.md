# sup7 reads mem7's token from the environment

- **Problem**: mem7 now requires a bearer token, and sup7 only took it as a literal in sup7.yaml.
- **Decision**: an empty `memory.token` falls back to `MEM7_TOKEN` from the environment.
- **Why**: the token stays in the service's environment file, shared with mesh7 and mem7, never in a YAML.
- **Where**: `src/sup7/config.py` (`MemoryConfig`).
