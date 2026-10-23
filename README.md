# Agent-WebCam

An agent project in its initial setup stage. The agent's purpose and runtime
have not been defined yet.

## Repository hygiene

- Keep source code, documentation, reviewed assets, and dependency lockfiles in Git.
- Keep secrets and machine-specific configuration local. Any `.env.example` must
  contain placeholders only.
- Store private camera captures, recordings, snapshots, and runtime data in the
  ignored `captures/`, `recordings/`, `snapshots/`, and `data/` directories.
- Use synthetic or explicitly approved, sanitized files for committed fixtures
  and documentation images.
- Keep generated files, caches, logs, and local agent state out of commits.
- Review staged filenames and contents before committing or publishing.

See [AGENTS.md](AGENTS.md) for contributor and coding-agent guidance.
