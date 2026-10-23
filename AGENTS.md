# Project instructions

## Keep the project simple

- Follow the requested scope. Ask: "Am I over-engineering this?"
- Prefer a small, understandable implementation and minimal dependencies.
- Keep source and documentation together in this repository. Do not commit
  temporary investigation output or duplicate generated files.

## Treat tracked files as public

- Never commit credentials, tokens, private keys, real `.env` values, private
  camera URLs, or machine-specific settings. Do not print secrets in logs.
- Use placeholders in example configuration.
- Put private camera media and runtime data in the ignored `captures/`,
  `recordings/`, `snapshots/`, or `data/` directories. Keep logs in `logs/`
  and temporary work in `tmp/`.
- Commit media and fixtures only after confirming they are synthetic or explicitly
  approved and sanitized. Check filenames, metadata, and embedded content too.
- Keep dependency lockfiles tracked. Exclude installed dependencies, caches,
  and build outputs.
- Do not force-add ignored content without a specific, reviewed reason.

## Verify changes

- Keep changes focused and documentation accurate.
- Run checks appropriate to the change once the implementation defines them.
- Before committing, run `git diff --check` and `git diff --cached --check`,
  inspect `git status --short`, and review `git diff --cached --stat`
  and `git diff --cached`.
- Ignore rules do not remove files already tracked. Check the tracked file list
  before publishing and report any accidental secret exposure immediately.
