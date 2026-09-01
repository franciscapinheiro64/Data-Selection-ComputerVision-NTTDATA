# uv Cheat Sheet

[uv](https://docs.astral.sh/uv/) is a fast Python package and project manager written in Rust. It replaces `pip`, `pip-tools`, `virtualenv`, and `pyenv` in a single tool.

## Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Project Commands

| Command | What It Does |
|---------|-------------|
| `uv init` | Create a new Python project (`pyproject.toml` + folder structure) |
| `uv init --name my_project` | Same but sets the project name explicitly |
| `uv sync` | Install all dependencies from `pyproject.toml` and create/update the `.venv` |
| `uv run main.py` | Run a script inside the project's virtual environment |
| `uv run pytest` | Run any command/tool inside the virtual environment |

## Dependency Management

| Command | What It Does |
|---------|-------------|
| `uv add <package>` | Add a dependency and install it (updates `pyproject.toml` + `uv.lock`) |
| `uv add "crewai>=0.100"` | Add with a version constraint |
| `uv add --dev pytest` | Add as a dev-only dependency |
| `uv remove <package>` | Remove a dependency (updates `pyproject.toml` + `uv.lock`) |
| `uv lock` | Regenerate `uv.lock` without installing anything |

## Python Version Management

| Command | What It Does |
|---------|-------------|
| `uv python install 3.12` | Download and install a Python version |
| `uv python list` | List available Python versions |
| `uv python pin 3.12` | Pin the project to a specific Python version (creates `.python-version`) |

## Virtual Environments

| Command | What It Does |
|---------|-------------|
| `uv venv` | Create a `.venv` in the current directory |
| `uv venv --python 3.12` | Create a `.venv` with a specific Python version |
| `source .venv/bin/activate` | Activate manually (not needed if you use `uv run`) |

## One-Off Scripts (no project needed)

| Command | What It Does |
|---------|-------------|
| `uv run --with requests script.py` | Run a script with a temporary dependency |
| `uvx ruff check .` | Run a CLI tool without installing it globally |

## Typical Workflow

```bash
uv init my_project        # 1. create project
cd my_project
uv add crewai python-dotenv  # 2. add dependencies
uv run main.py            # 3. run
```

## Key Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project metadata and dependencies |
| `uv.lock` | Locked dependency versions (commit this to git) |
| `.python-version` | Pinned Python version (optional) |
| `.venv/` | Virtual environment (do not commit — add to `.gitignore`) |

## Boas práticas do repositório

- Lint/format: `ruff`. Type checking: `mypy`. Testes: `pytest`.
- `pre-commit` corre estas verificações antes de cada commit.
- CI (GitHub Actions) corre lint + testes em cada pull request.
- Conventional Commits para mensagens de commit — facilita histórico e changelog.
- Branch protection na `main`: PR obrigatório, checks verdes obrigatórios.
