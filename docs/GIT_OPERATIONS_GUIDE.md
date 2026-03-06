# Comprehensive Git Operations Guide

A detailed technical reference for Git version control operations, synchronization workflows, and VS Code Copilot integration.

---

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Repository Architecture](#repository-architecture)
3. [Fundamental Operations](#fundamental-operations)
4. [Synchronization Workflows](#synchronization-workflows)
5. [Branch Management](#branch-management)
6. [Merge Operations](#merge-operations)
7. [VS Code & Copilot Integration](#vs-code--copilot-integration)
8. [Conflict Resolution](#conflict-resolution)
9. [Best Practices](#best-practices)
10. [Command Reference](#command-reference)

---

## Core Concepts

### What is Git?

Git is a distributed version control system (DVCS). Unlike centralized systems (SVN, CVS), every Git clone is a full repository with complete history and version-tracking capabilities, independent of network access or a central server.

### Key Terminology

| Term | Definition |
|------|------------|
| **Repository (repo)** | A directory containing your project files and the entire history of changes in a hidden `.git` folder |
| **Commit** | A snapshot of your repository at a specific point in time, identified by a SHA-1 hash (e.g., `a7cc0fb`) |
| **Branch** | A lightweight movable pointer to a commit; allows parallel development |
| **HEAD** | A pointer to the current branch reference (which commit you're working on) |
| **Working Directory** | The actual files on your disk that you edit |
| **Staging Area (Index)** | A file in `.git` that stores information about what will go into your next commit |
| **Remote** | A version of your repository hosted on a server (e.g., GitHub, GitLab, Bitbucket) |
| **Origin** | The default name for the remote repository you cloned from |
| **Upstream** | Typically refers to the original repository when you've forked someone else's project |

### The Three States of Git

Files in Git exist in three main states:

```
┌─────────────────┐    git add    ┌─────────────────┐   git commit   ┌─────────────────┐
│    MODIFIED     │ ───────────→  │     STAGED      │ ─────────────→ │    COMMITTED    │
│ (Working Dir)   │               │  (Staging Area) │                │   (.git repo)   │
└─────────────────┘               └─────────────────┘                └─────────────────┘
        ↑                                                                     │
        └─────────────────────── git checkout ────────────────────────────────┘
```

1. **Modified**: You've changed the file but haven't staged it
2. **Staged**: You've marked a modified file to go into your next commit
3. **Committed**: Data is safely stored in your local database

---

## Repository Architecture

### Local vs Remote

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              REMOTE (GitHub)                               │
│                                                                            │
│  origin/master ──── origin/feature-1 ──── origin/feature-2                │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                                    ↑
                               NETWORK
                          (push / pull / fetch)
                                    ↓
┌────────────────────────────────────────────────────────────────────────────┐
│                               LOCAL MACHINE                                │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                         .git Repository                              │ │
│  │                                                                      │ │
│  │  Local Branches:     master, feature-1, feature-2                   │ │
│  │  Remote-Tracking:    origin/master, origin/feature-1, etc.          │ │
│  │  Objects:            commits, trees, blobs (file contents)          │ │
│  │  References:         HEAD, tags, branch pointers                    │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                         Working Directory                            │ │
│  │                                                                      │ │
│  │  Your actual project files (what you edit in VS Code)               │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                         Staging Area (Index)                         │ │
│  │                                                                      │ │
│  │  Files prepared for next commit (.git/index)                        │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Remote-Tracking Branches

Remote-tracking branches (e.g., `origin/master`) are local references to the state of remote branches. They:

- Are updated when you `fetch` or `pull`
- Cannot be directly modified (read-only locally)
- Allow you to see what the remote looked like last time you connected
- Follow the naming convention: `<remote>/<branch>`

---

## Fundamental Operations

### git clone

**Purpose**: Create a local copy of a remote repository.

```powershell
git clone https://github.com/user/repo.git
git clone git@github.com:user/repo.git              # SSH
git clone https://github.com/user/repo.git mydir    # Clone into specific directory
git clone --depth 1 <url>                           # Shallow clone (latest commit only)
git clone --branch dev <url>                        # Clone specific branch
```

**What happens internally**:
1. Creates a new directory
2. Initializes `.git` inside it
3. Creates remote-tracking branches for each branch in the cloned repository
4. Creates and checks out the default branch (usually `master` or `main`)

---

### git fetch

**Purpose**: Download commits, files, and refs from a remote repository into your local repo. Does NOT modify your working directory.

```powershell
git fetch                       # Fetch from origin (default remote)
git fetch origin                # Explicit origin fetch
git fetch --all                 # Fetch from all remotes
git fetch origin master         # Fetch only master branch
git fetch --prune               # Remove remote-tracking refs that no longer exist on remote
git fetch --tags                # Fetch all tags
```

**What happens internally**:
1. Contacts the remote repository
2. Downloads any commits and data you don't have
3. Updates your remote-tracking branches (e.g., `origin/master`)
4. Does NOT update your local branches or working directory

**Use case**: See what others have done without affecting your work.

```powershell
git fetch
git log HEAD..origin/master     # See commits on remote you don't have
git diff master origin/master   # See file differences
```

---

### git pull

**Purpose**: Fetch from remote AND integrate changes into your current branch. Essentially `git fetch` + `git merge` (or `git rebase`).

```powershell
git pull                        # Fetch + merge from tracking branch
git pull origin master          # Pull specific branch
git pull --rebase               # Fetch + rebase instead of merge
git pull --no-commit            # Merge but don't auto-commit (allows inspection)
git pull --ff-only              # Only pull if fast-forward is possible
```

**Merge vs Rebase pull**:

```
# git pull (merge - default)
Before:        A---B---C  (master)
                    \
                     D---E  (origin/master)

After:         A---B---C---F  (master, merge commit)
                    \     /
                     D---E  (origin/master)

# git pull --rebase
Before:        A---B---C  (master)
                    \
                     D---E  (origin/master)

After:         A---B---D---E---C'  (master, C rebased onto E)
```

**Configuration for default behavior**:
```powershell
git config --global pull.rebase false   # Default: merge
git config --global pull.rebase true    # Always rebase on pull
git config --global pull.ff only        # Only fast-forward
```

---

### git push

**Purpose**: Upload local branch commits to a remote repository.

```powershell
git push                        # Push current branch to its upstream
git push origin master          # Push master to origin
git push -u origin feature      # Push and set upstream tracking
git push --force                # Force push (DANGEROUS - overwrites remote history)
git push --force-with-lease     # Safer force push (fails if remote has new commits)
git push --all                  # Push all local branches
git push --tags                 # Push all tags
git push origin --delete branch # Delete remote branch
```

**Setting upstream tracking**:
```powershell
# First push of a new branch
git push -u origin feature-branch
# or
git push --set-upstream origin feature-branch

# After this, you can just use:
git push
```

**What happens internally**:
1. Git compares your local branch with the remote branch
2. Calculates which commits need to be sent
3. Sends commit objects to the remote
4. Updates the remote branch reference

**Push rejection scenarios**:
```
! [rejected]        master -> master (non-fast-forward)
```
This means remote has commits you don't have. Solution:
```powershell
git pull
# Resolve any conflicts
git push
```

---

### git add

**Purpose**: Add file contents to the staging area (index) for the next commit.

```powershell
git add file.txt                # Stage specific file
git add .                       # Stage all changes in current directory and subdirs
git add -A                      # Stage all changes (including deletions) in entire repo
git add -p                      # Interactive staging (choose hunks)
git add *.py                    # Stage all Python files
git add src/                    # Stage entire directory
```

**Staging area inspection**:
```powershell
git status                      # See staged vs unstaged
git diff --staged               # See what's staged
git diff --cached               # Same as --staged
```

**Unstaging files**:
```powershell
git reset HEAD file.txt         # Unstage file (keep changes)
git restore --staged file.txt   # Modern alternative (Git 2.23+)
```

---

### git commit

**Purpose**: Record changes to the repository by creating a new commit.

```powershell
git commit -m "Commit message"              # Commit with message
git commit -am "Message"                    # Stage all tracked files and commit
git commit --amend                          # Modify the last commit
git commit --amend -m "New message"         # Change last commit message
git commit --amend --no-edit                # Add staged changes to last commit, keep message
git commit --allow-empty -m "Empty commit"  # Create commit with no changes
```

**Good commit message format**:
```
<type>: <short summary> (50 chars or less)

<body - detailed explanation if needed>
(wrap at 72 characters)

<footer - references, breaking changes>
```

**Types**: feat, fix, docs, style, refactor, test, chore

**Example**:
```
feat: Add user authentication module

Implements JWT-based authentication with refresh tokens.
Includes middleware for protected routes.

Closes #123
```

---

### git checkout

**Purpose**: Switch branches or restore working tree files.

```powershell
git checkout master                 # Switch to master branch
git checkout -b new-branch          # Create and switch to new branch
git checkout -b feature origin/feature  # Create local branch from remote
git checkout -- file.txt            # Discard changes in working directory
git checkout HEAD~2 -- file.txt     # Restore file from 2 commits ago
git checkout abc123                 # Checkout specific commit (detached HEAD)
```

**Modern alternatives (Git 2.23+)**:
```powershell
git switch master                   # Switch branches
git switch -c new-branch            # Create and switch
git restore file.txt                # Discard working directory changes
git restore --staged file.txt       # Unstage file
```

**Detached HEAD state**:
When you checkout a commit directly (not a branch), you're in "detached HEAD" state:
```powershell
git checkout abc1234
# Warning: You are in 'detached HEAD' state...
```
Changes made here won't belong to any branch unless you create one:
```powershell
git checkout -b new-branch-from-here
```

---

### git merge

**Purpose**: Join two or more development histories together.

```powershell
git merge feature-branch            # Merge feature-branch into current branch
git merge --no-ff feature           # Force merge commit even if fast-forward possible
git merge --squash feature          # Squash all commits into one (doesn't auto-commit)
git merge --abort                   # Abort merge in progress (during conflict)
git merge -X ours feature           # Merge, preferring our changes on conflict
git merge -X theirs feature         # Merge, preferring their changes on conflict
```

**Merge strategies**:

1. **Fast-forward**: When current branch has no new commits since feature branched
```
Before:     A---B  (master)
                 \
                  C---D  (feature)

After:      A---B---C---D  (master, feature)
```

2. **Three-way merge**: When both branches have new commits
```
Before:     A---B---E  (master)
                 \
                  C---D  (feature)

After:      A---B---E---F  (master, merge commit)
                 \     /
                  C---D
```

3. **Squash merge**: Combine all feature commits into one
```powershell
git merge --squash feature
git commit -m "Feature: complete implementation"
```

---

### git rebase

**Purpose**: Reapply commits on top of another base tip. Creates a linear history.

```powershell
git rebase master                   # Rebase current branch onto master
git rebase -i HEAD~3                # Interactive rebase (edit last 3 commits)
git rebase --onto main A B          # Rebase commits from A to B onto main
git rebase --abort                  # Abort rebase in progress
git rebase --continue               # Continue after resolving conflicts
git rebase --skip                   # Skip current commit during rebase
```

**Merge vs Rebase**:
```
# Merge: preserves history as it happened
      A---B---C  (feature)
     /         \
D---E---F---G---H  (master)

# Rebase: creates linear history
D---E---F---G---A'---B'---C'  (master with rebased feature)
```

**Interactive rebase options** (`git rebase -i`):
- `pick` - use commit as-is
- `reword` - use commit, but edit message
- `edit` - use commit, but stop to amend
- `squash` - meld into previous commit
- `fixup` - like squash but discard message
- `drop` - remove commit

**⚠️ WARNING**: Never rebase commits that have been pushed to a shared repository. Rebase rewrites history.

---

### git stash

**Purpose**: Temporarily store modified tracked files to work on something else.

```powershell
git stash                           # Stash changes
git stash push -m "description"     # Stash with message
git stash list                      # List stashes
git stash pop                       # Apply most recent stash and remove it
git stash apply                     # Apply most recent stash (keep in stash list)
git stash apply stash@{2}           # Apply specific stash
git stash drop stash@{0}            # Delete specific stash
git stash clear                     # Delete all stashes
git stash show -p                   # Show stash diff
git stash branch new-branch         # Create branch from stash
```

---

### git reset

**Purpose**: Reset current HEAD to a specified state.

```powershell
git reset --soft HEAD~1             # Undo last commit, keep changes staged
git reset --mixed HEAD~1            # Undo last commit, keep changes unstaged (default)
git reset --hard HEAD~1             # Undo last commit, DISCARD all changes
git reset HEAD file.txt             # Unstage a file
git reset --hard origin/master      # Reset to match remote exactly
```

**Reset modes explained**:
```
                     HEAD    Index   Working Dir
--soft               YES     NO      NO
--mixed (default)    YES     YES     NO
--hard               YES     YES     YES
```

**⚠️ WARNING**: `git reset --hard` is destructive. Uncommitted changes are lost permanently.

---

### git revert

**Purpose**: Create a new commit that undoes changes from a previous commit. Safe for shared history.

```powershell
git revert HEAD                     # Revert last commit
git revert abc1234                  # Revert specific commit
git revert HEAD~3..HEAD             # Revert last 3 commits
git revert -n HEAD                  # Revert without auto-commit
git revert -m 1 <merge-commit>      # Revert a merge commit
```

**Reset vs Revert**:
- `reset`: Rewrites history (dangerous for shared branches)
- `revert`: Creates new commit (safe for shared branches)

---

## Synchronization Workflows

### Standard Sync Workflow

```powershell
# 1. Save current work
git stash                           # If you have uncommitted changes

# 2. Get latest from remote
git fetch origin

# 3. Check status
git status
git log HEAD..origin/master --oneline   # Commits you're missing

# 4. Integrate changes
git pull                            # Or: git merge origin/master

# 5. Restore your work
git stash pop                       # If you stashed earlier

# 6. Push your changes
git push
```

### Feature Branch Workflow

```powershell
# 1. Start from updated master
git checkout master
git pull

# 2. Create feature branch
git checkout -b feature/my-feature

# 3. Make changes, commit often
git add .
git commit -m "feat: implement feature"

# 4. Keep feature branch updated with master
git fetch origin
git rebase origin/master            # Or: git merge origin/master

# 5. Push feature branch
git push -u origin feature/my-feature

# 6. Create Pull Request on GitHub

# 7. After PR is merged, clean up
git checkout master
git pull
git branch -d feature/my-feature    # Delete local branch
git push origin --delete feature/my-feature  # Delete remote branch
```

### Handling Divergent Branches

When your local and remote have diverged:

```
Local:     A---B---C  (master)
Remote:    A---B---D---E  (origin/master)
```

**Option 1: Merge (preserves all history)**
```powershell
git pull
# Creates merge commit
git push
```

**Option 2: Rebase (linear history)**
```powershell
git pull --rebase
# Replays C on top of E
git push
```

**Option 3: Force push (DANGEROUS - only if you own the branch)**
```powershell
git push --force-with-lease
```

---

## Branch Management

### Viewing Branches

```powershell
git branch                          # List local branches
git branch -r                       # List remote branches
git branch -a                       # List all branches
git branch -v                       # List with last commit
git branch -vv                      # List with tracking info
git branch --merged                 # Branches merged into current
git branch --no-merged              # Branches not merged
```

### Creating Branches

```powershell
git branch new-branch               # Create branch (don't switch)
git checkout -b new-branch          # Create and switch
git checkout -b feature origin/feature  # Create from remote branch
git branch new-branch abc1234       # Create from specific commit
```

### Deleting Branches

```powershell
git branch -d branch-name           # Delete (only if merged)
git branch -D branch-name           # Force delete
git push origin --delete branch     # Delete remote branch
git remote prune origin             # Remove stale remote-tracking branches
```

### Renaming Branches

```powershell
git branch -m old-name new-name     # Rename local branch
git branch -m new-name              # Rename current branch

# Update remote
git push origin --delete old-name
git push -u origin new-name
```

### Tracking Branches

```powershell
git branch -u origin/feature        # Set upstream for current branch
git branch --set-upstream-to=origin/feature feature
git checkout --track origin/feature # Create local tracking branch
```

---

## VS Code & Copilot Integration

### VS Code Source Control Panel

Located in the sidebar (Ctrl+Shift+G), provides GUI for:

| Action | Equivalent Command |
|--------|-------------------|
| Stage Changes (+) | `git add <file>` |
| Unstage Changes (-) | `git reset HEAD <file>` |
| Discard Changes | `git checkout -- <file>` |
| Commit (✓) | `git commit -m "message"` |
| Sync (↑↓) | `git pull && git push` |
| Pull (↓) | `git pull` |
| Push (↑) | `git push` |

### VS Code Status Bar

Bottom left shows:
- Current branch name
- Sync status (↑ commits to push, ↓ commits to pull)
- Click to switch branches or sync

### Copilot Chat Buttons

When using GitHub Copilot in cloud/agent mode:

| Button | What It Does | Equivalent Commands |
|--------|--------------|---------------------|
| **Checkout** | Downloads and switches to a branch created by Copilot | `git fetch origin <branch> && git checkout <branch>` |
| **Apply** | Applies a specific code change to your local file | Similar to `git apply` or `git cherry-pick` for a single change |

**When these appear**:
- Copilot creates a branch on GitHub (remote) via agent mode
- The Checkout button pulls that branch locally
- Apply extracts just the code diff without switching branches

**Local VS Code Chat (Agent Mode)**:
- Changes are made directly to your local files
- No Checkout/Apply buttons needed
- You still need to commit and push manually

### VS Code Merge Editor

When conflicts occur:
1. VS Code highlights conflicting files
2. Click file to open 3-way merge editor
3. Choose: **Accept Current** | **Accept Incoming** | **Accept Both**
4. Or manually edit the result
5. Stage the resolved file
6. Commit

---

## Conflict Resolution

### What Causes Conflicts

Conflicts occur when:
1. Two branches modify the same line(s) of the same file
2. One branch deletes a file another branch modified
3. Both branches add a file with the same name

### Conflict Markers

```
<<<<<<< HEAD
Your changes (current branch)
=======
Their changes (incoming branch)
>>>>>>> feature-branch
```

Sometimes with 3-way markers:
```
<<<<<<< HEAD
Your changes
||||||| base
Original content
=======
Their changes
>>>>>>> feature-branch
```

### Resolution Process

```powershell
# 1. Start merge/pull/rebase
git merge feature-branch

# 2. See conflicting files
git status
# "both modified: file.txt"

# 3. Open file and resolve (remove markers, keep correct code)

# 4. Mark as resolved
git add file.txt

# 5. Complete merge
git commit                          # Opens editor
# or
git commit -m "Resolve merge conflicts"

# 6. If things go wrong
git merge --abort                   # Cancel the merge
```

### Resolution Strategies

**For entire files**:
```powershell
git checkout --ours file.txt        # Keep your version
git checkout --theirs file.txt      # Keep their version
```

**For merge command**:
```powershell
git merge -X ours feature           # Auto-resolve preferring ours
git merge -X theirs feature         # Auto-resolve preferring theirs
```

### Preventing Conflicts

1. **Pull frequently**: Stay up to date with `master`
2. **Small commits**: Easier to merge
3. **Communicate**: Coordinate who's working on what
4. **Feature branches**: Isolate work
5. **Rebase regularly**: `git rebase master` on feature branches

---

## Best Practices

### Commit Discipline

1. **Atomic commits**: One logical change per commit
2. **Meaningful messages**: Describe what and why
3. **Test before committing**: Don't commit broken code
4. **Don't commit generated files**: Use `.gitignore`

### Branch Hygiene

1. **Never commit directly to master**: Use feature branches
2. **Delete merged branches**: Keep repo clean
3. **Use descriptive names**: `feature/user-auth`, `fix/login-bug`
4. **Keep branches short-lived**: Merge frequently

### Sync Strategy

1. **Pull before starting work**: `git pull`
2. **Pull before pushing**: `git pull` then `git push`
3. **Fetch to inspect**: `git fetch` to see without merging
4. **Don't force push shared branches**: Unless you coordinate

### Security

1. **Never commit secrets**: Use environment variables
2. **Use `.gitignore`**: Exclude sensitive files
3. **Review diffs before committing**: `git diff --staged`
4. **Sign commits**: `git commit -S`

---

## Command Reference

### Quick Reference Table

| Task | Command |
|------|---------|
| Clone repo | `git clone <url>` |
| Check status | `git status` |
| Stage all changes | `git add .` |
| Commit | `git commit -m "message"` |
| Push | `git push` |
| Pull | `git pull` |
| Create branch | `git checkout -b <name>` |
| Switch branch | `git checkout <name>` |
| Merge branch | `git merge <branch>` |
| View log | `git log --oneline` |
| Discard changes | `git checkout -- <file>` |
| Undo last commit | `git reset --soft HEAD~1` |
| Stash changes | `git stash` |
| Apply stash | `git stash pop` |

### Useful Aliases

Add to `~/.gitconfig` or run `git config --global alias.<name> "<command>"`:

```ini
[alias]
    co = checkout
    br = branch
    ci = commit
    st = status
    unstage = reset HEAD --
    last = log -1 HEAD
    visual = log --oneline --graph --all --decorate
    amend = commit --amend --no-edit
    undo = reset --soft HEAD~1
    sync = !git pull && git push
```

### Inspection Commands

```powershell
git log --oneline --graph --all     # Visual branch history
git log -p                          # Show patches
git log --stat                      # Show files changed
git show <commit>                   # Show commit details
git diff                            # Working dir vs staging
git diff --staged                   # Staging vs last commit
git diff branch1..branch2           # Compare branches
git blame file.txt                  # Who changed each line
git reflog                          # History of HEAD movements
```

### Configuration

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@email.com"
git config --global core.editor "code --wait"   # VS Code as editor
git config --global init.defaultBranch main     # Default branch name
git config --global pull.rebase false           # Merge on pull
git config --list                               # View all config
```

---

## Troubleshooting

### "Your branch has diverged"

```powershell
git pull                            # Merge remote changes
# or
git pull --rebase                   # Rebase your changes
```

### "Permission denied (publickey)"

```powershell
# Check if SSH key exists
ls ~/.ssh/id_*

# Generate if needed
ssh-keygen -t ed25519 -C "your@email.com"

# Add to SSH agent
ssh-add ~/.ssh/id_ed25519

# Add public key to GitHub Settings > SSH keys
cat ~/.ssh/id_ed25519.pub
```

### "Failed to push some refs"

```powershell
git pull                            # Get remote changes first
# Resolve conflicts if any
git push
```

### Recover deleted branch

```powershell
git reflog                          # Find the commit hash
git checkout -b recovered-branch <hash>
```

### Undo a pushed commit

```powershell
git revert <commit-hash>            # Creates new undo commit
git push
```

---

## Summary

**The Golden Workflow**:
```
git pull → edit → git add → git commit → git pull → git push
```

**Key Principles**:
1. Always pull before starting work
2. Commit early, commit often
3. Pull before pushing
4. Use feature branches
5. Write meaningful commit messages
6. Never force push shared branches

---

*Document version: 1.0*
*Last updated: March 4, 2026*
