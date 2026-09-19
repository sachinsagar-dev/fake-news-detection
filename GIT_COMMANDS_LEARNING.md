# Git Commands Used in the Fake News Detection Project

This file explains the Git commands used while synchronizing the local project with the GitHub repository, including the file-mismatch problem and how it was solved.

---

## 1. Local Git vs GitHub

There are three important places:

```text
Your project folder
        |
        | git add / git commit
        v
Local Git repository
        |
        | git push / git fetch / git pull
        v
GitHub repository
```

- **Working directory**: your actual project files.
- **Local repository**: Git history stored inside the `.git` folder.
- **Remote repository**: GitHub, called `origin` in this project.

---

## 2. `git status`

```bash
git status
```

Shows the current state of the repository.

It tells you:
- current branch
- whether the branch is ahead/behind the remote
- modified files
- staged files
- untracked files

In our case:

```text
Your branch is behind 'origin/main' by 2 commits.

Untracked files:
    artifacts/error_analysis/
    error_analysis.py
```

An **untracked file** is visible to Git but is not currently tracked.

---

## 3. `git diff`

```bash
git diff
```

Shows unstaged changes in **tracked files**.

It showed nothing in our case because there were no tracked-file modifications.

Important:

> `git diff` does not normally show the contents of an untracked file.

---

## 4. `git diff --stat`

```bash
git diff --stat
```

Shows a compact summary of tracked-file changes.

It also showed nothing because there were no tracked-file modifications.

---

## 5. `git log --oneline -4`

```bash
git log --oneline -4
```

Shows recent commits in compact form.

We saw:

```text
2d2d876 (HEAD -> main) intitial project setup
```

This showed that our local `main` was still on the older commit.

- `2d2d876` = short commit SHA
- `HEAD` = what the current checkout points to
- `main` = local branch

---

## 6. `git fetch origin`

```bash
git fetch origin
```

Downloads information about commits and branches from GitHub **without changing your checked-out files**.

`origin` is the usual name of the GitHub remote.

You can inspect remotes with:

```bash
git remote -v
```

Our output contained:

```text
2d2d876..bcf5d5e  main -> origin/main
```

This meant Git learned that remote `main` had moved from `2d2d876` to `bcf5d5e`.

Think:

> `git fetch` = "Tell my local Git what changed on GitHub."

---

## 7. Why `git pull --ff-only` failed

```bash
git pull --ff-only origin main
```

`git pull` fetches remote changes and then integrates them into the current branch.

The `--ff-only` option allows the update only when it can move the branch forward without creating a merge commit.

It failed with:

```text
error: The following untracked working tree files would be overwritten by merge:
        error_analysis.py
Please move or remove them before you merge.
```

### What happened?

Locally we had an **untracked**:

```text
error_analysis.py
```

GitHub also contained a **tracked**:

```text
error_analysis.py
```

Git refused to continue because the incoming file would overwrite the local untracked file.

This was Git protecting local work.

---

## 8. How we solved the mismatch

We did not delete the local file.

We renamed it:

```bash
mv error_analysis.py error_analysis_local_backup.py
```

`mv` means move/rename.

Now the original filename was free:

```text
error_analysis_local_backup.py   <- old local copy
error_analysis.py                <- can now come from GitHub
```

This was safer than deleting the file.

---

## 9. Pulling again

We then ran:

```bash
git pull --ff-only origin main
```

It succeeded:

```text
Fast-forward
 README.md         |   2 +-
 error_analysis.py | 121 ++++++++++++++++++++++++++++++++++++++++++++++++++++++
```

### What does fast-forward mean?

The remote branch was simply ahead of the local branch, so Git moved the local branch pointer forward.

No merge commit was needed.

---

## 10. Comparing the two files with `diff -u`

```bash
diff -u error_analysis.py error_analysis_local_backup.py
```

Compares two files line-by-line.

The `-u` option means unified diff format.

The result showed that the two versions were not identical, although both implemented error analysis.

This confirmed that we should not blindly assume the local file and GitHub file were the same.

---

## 11. Why we kept the GitHub version

After the successful pull:

- `error_analysis.py` = repository version
- `error_analysis_local_backup.py` = old local copy

The repository version was already committed to GitHub, so it became the canonical project version.

The backup was preserved until we were comfortable removing it.

---

## 12. Final `git status`

We checked:

```bash
git status
```

and saw:

```text
Your branch is up to date with 'origin/main'.
```

That confirmed local `main` was synchronized with GitHub.

The only remaining entries were untracked local/generated files.

---

## 13. `.gitignore` and generated ML files

The project uses rules such as:

```gitignore
artifacts/*.json
artifacts/*.csv
artifacts/*.png
```

So generated outputs such as:

```text
artifacts/error_analysis/false_positives.csv
artifacts/error_analysis/false_negatives.csv
artifacts/error_analysis/error_summary.json
```

are intentionally ignored.

This keeps generated outputs out of the source history unless we explicitly decide to version them.

---

## 14. Quick Git reference

| Command | Purpose |
|---|---|
| `git status` | Show repository state |
| `git diff` | Show unstaged changes to tracked files |
| `git diff --stat` | Summarize tracked-file changes |
| `git log --oneline -4` | Show recent commits |
| `git fetch origin` | Download remote information without changing working files |
| `git pull --ff-only origin main` | Fetch and fast-forward local `main` |
| `mv old new` | Rename/move a file |
| `diff -u file1 file2` | Compare two files |
| `git remote -v` | Show configured remotes |

---

## 15. Normal Git workflow

For normal development:

```bash
git status
git add .
git commit -m "Describe the change"
git push origin main
```

When GitHub may contain new commits:

```bash
git fetch origin
git status
git pull --ff-only origin main
```

---

## 16. `git fetch` vs `git pull`

### `git fetch`

Downloads remote information but does not integrate it into your current branch.

```bash
git fetch origin
```

Think:

> "Tell me what changed on GitHub."

### `git pull`

Fetches remote changes and integrates them into your current branch.

```bash
git pull origin main
```

Think:

> "Bring the GitHub changes into my local branch."

---

## 17. `git push` vs `git pull`

Direction:

```text
LOCAL  -- git push -->  GITHUB
LOCAL  <-- git pull --  GITHUB
```

- `push` sends your commits to GitHub.
- `pull` brings remote changes into your local branch.

---

## 18. Our exact real-project sequence

```bash
git fetch origin
git status
git diff
git diff --stat
git pull --ff-only origin main
```

The pull was blocked because a local untracked `error_analysis.py` would have been overwritten by GitHub's tracked `error_analysis.py`.

We then used:

```bash
mv error_analysis.py error_analysis_local_backup.py
git pull --ff-only origin main
diff -u error_analysis.py error_analysis_local_backup.py
git status
```

The repository was then synchronized successfully.

---

## 19. Interview explanation

A strong interview explanation is:

> "I had a local untracked `error_analysis.py` while the same file had already been committed to the remote repository. When I ran `git pull --ff-only`, Git prevented the pull because the incoming tracked file would overwrite my local untracked file. I preserved the local copy by renaming it, pulled the remote changes using a fast-forward-only pull, compared both versions with `diff -u`, and then kept the repository version as the canonical version."

This demonstrates understanding of Git's behavior, not just memorized commands.
