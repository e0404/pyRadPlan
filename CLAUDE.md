# Claude Guide — pyRadPlan

See [AGENTS.md](AGENTS.md) for project conventions, setup, branching, code style, testing, and PR checklist.

## Extra Notes for Claude

- Prefer editing existing files over creating new ones.
- Do not add comments unless the reason is non-obvious from the code itself.
- Keep new numerical code array-backend-agnostic (numpy/cupy/torch via `array_api_compat`).
- When touching a dose engine or geometry module, run the test suite before reporting done.
- Do not push to remote or open PRs without explicit user confirmation.
