# Contributing Guide

Welcome! This document explains how we collaborate on this project. Please read it before pushing any changes.

---

## Table of Contents

- [Contributing Guide](#contributing-guide)
  - [Table of Contents](#table-of-contents)
  - [1. Never push directly to `main`](#1-never-push-directly-to-main)
  - [2. Branch naming](#2-branch-naming)
  - [3. Commit messages](#3-commit-messages)
  - [4. Opening a Pull Request (PR)](#4-opening-a-pull-request-pr)
    - [Step-by-step](#step-by-step)
    - [PR title](#pr-title)
    - [PR description](#pr-description)
    - [Example PR description](#example-pr-description)
    - [Keep PRs small](#keep-prs-small)
  - [5. Code review etiquette](#5-code-review-etiquette)
  - [6. Linting before committing](#6-linting-before-committing)

---

## 1. Never push directly to `main`

All changes must go through a Pull Request. Direct pushes to `main` skip review, break the CI pipeline, and make history hard to follow.

```bash
# ❌ Never do this
git checkout main
git commit -am "fix stuff"
git push origin main

# ✅ Always work on a feature branch (see section 2)
```

---

## 2. Branch naming

Use the pattern `<type>/<short-description>`. Keep it lowercase with hyphens.

| Type | When to use | Example |
|------|-------------|---------|
| `feat/` | Adding new functionality | `feat/clip-embeddings` |
| `fix/` | Fixing a bug | `fix/kmeans-nan-crash` |
| `refactor/` | Restructuring code without changing behaviour | `refactor/extract-cluster-utils` |
| `docs/` | Documentation only | `docs/update-readme` |
| `chore/` | Tooling, CI, dependencies | `chore/add-ruff-action` |

```bash
# Create and switch to a new branch
git checkout -b feat/dinov2-embeddings
```

---

## 3. Commit messages

Write short, imperative messages that explain *what* and *why*, not *how*.

```
# ✅ Good
Add DINOv2 feature extractor
Fix KeyError when image path has spaces
Remove unused import in clustering.py

# ❌ Bad
changes
WIP
fixed bug
asdfgh
```

For larger commits, add a body after a blank line:

```
Add DINOv2 feature extractor

Uses the facebook/dinov2-base model via HuggingFace. Chosen over CLIP
because it produces richer spatial features for fine-grained clustering.
```

---

## 4. Opening a Pull Request (PR)

### Step-by-step

```bash
# 1. Make sure you are on your feature branch
git checkout feat/dinov2-embeddings

# 2. Pull the latest main to avoid conflicts
git fetch origin
git merge origin/main

# 3. Push your branch
git push origin feat/dinov2-embeddings

# 4. Open the PR on GitHub:
#    Repository → Pull Requests → New Pull Request
#    base: main  ←  compare: feat/dinov2-embeddings
```

### PR title

Same convention as commit messages — short and imperative:

```
# ✅ Good
Add DINOv2 feature extractor
Fix KeyError when image path has spaces

# ❌ Bad
My changes
WIP branch
```

### PR description

Always fill in the description. A minimal template:

```markdown
## What
Brief summary of what this PR does.

## Why
Why is this change needed?

## How to test
Steps to verify the change works (e.g., run a script, check an output file).
```

### Example PR description

```markdown
## What
Adds a DINOv2 backbone as an alternative to CLIP for generating image embeddings.

## Why
DINOv2 produces richer spatial features, which improves cluster separation on
fine-grained datasets compared to CLIP's global token.

## How to test
1. Run `python embed.py --backbone dinov2 --dataset samples/`
2. Check that `embeddings.npy` is created with shape `(N, 768)`.
3. Run `python cluster.py` and inspect the output clusters visually.
```

### Keep PRs small

One PR = one logical change. Small PRs are reviewed faster and are easier to revert if something goes wrong.

```
# ✅ One focused PR
feat/add-dinov2-extractor   → only adds the new backbone

# ❌ Too much in one PR
feat/big-refactor           → new backbone + clustering changes + new CLI + docs
```

---

## 5. Code review etiquette

- **Author:** respond to every comment, either by making a change or explaining why you disagree.
- **Reviewer:** be specific. Instead of "this is wrong", write "this will crash when `embeddings` is empty — add a guard".
- A PR needs **at least one approval** before merging.
- Use the **"Request changes"** option if something must be fixed before merging.
- Prefer **squash merge** to keep `main` history clean.

---

## 6. Linting before committing

This project uses [Ruff](https://docs.astral.sh/ruff/). Install the pre-commit hook once:

```bash
pip install pre-commit
pre-commit install
```

After that, ruff runs automatically before every commit and blocks it if there are errors. See the [README](README.md#ci--linting) for more details.

To run manually at any time:

```bash
pre-commit run --all-files
```
