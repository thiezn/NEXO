use git2::{Cred, FetchOptions, ObjectType, PushOptions, RemoteCallbacks, Repository, Signature};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Git-backed file storage for NEXO persistent data.
///
/// Wraps a `git2::Repository` behind a `Mutex` for thread safety. All operations
/// are synchronous (git2 is blocking); callers should use `tokio::task::spawn_blocking`.
pub struct GitStorage {
    repo: Mutex<Repository>,
    repo_path: PathBuf,
}

// SAFETY: git2::Repository is not Send/Sync but we guard it with a Mutex and
// only access it through the Mutex, making the wrapper safe to share.
unsafe impl Send for GitStorage {}
unsafe impl Sync for GitStorage {}

impl GitStorage {
    /// Open an existing git repository at `path`.
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let repo = Repository::open(path)?;
        Ok(Self {
            repo: Mutex::new(repo),
            repo_path: path.to_path_buf(),
        })
    }

    /// Pull latest changes from `origin/main`. Fast-forward only.
    /// No-op if the remote has no branches yet (empty repo).
    pub fn pull(&self) -> anyhow::Result<()> {
        let repo = self
            .repo
            .lock()
            .map_err(|e| anyhow::anyhow!("mutex poisoned: {e}"))?;
        pull_impl(&repo)
    }

    /// Read a file from the working tree.
    pub fn read_file(&self, rel_path: &str) -> anyhow::Result<String> {
        let full = self.repo_path.join(rel_path);
        Ok(std::fs::read_to_string(&full)?)
    }

    /// Check whether a file exists in the working tree.
    pub fn file_exists(&self, rel_path: &str) -> bool {
        self.repo_path.join(rel_path).exists()
    }

    /// List files under a directory prefix.
    /// Returns filenames relative to the prefix (e.g., for prefix "NOTES/",
    /// a file at "NOTES/foo.md" is returned as "foo.md").
    pub fn list_files(&self, prefix: &str) -> anyhow::Result<Vec<String>> {
        let dir = self.repo_path.join(prefix);
        if !dir.is_dir() {
            return Ok(Vec::new());
        }
        let mut files = Vec::new();
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let ft = entry.file_type()?;
            if ft.is_file() {
                if let Some(name) = entry.file_name().to_str() {
                    files.push(name.to_string());
                }
            }
        }
        files.sort();
        Ok(files)
    }

    /// Write a file, commit, and push to origin.
    ///
    /// Sequence: pull → mkdir_p → write file → add → commit → push.
    pub fn write_and_sync(
        &self,
        rel_path: &str,
        content: &str,
        commit_msg: &str,
    ) -> anyhow::Result<()> {
        let repo = self
            .repo
            .lock()
            .map_err(|e| anyhow::anyhow!("mutex poisoned: {e}"))?;

        // Pull first to avoid divergence
        if let Err(e) = pull_impl(&repo) {
            tracing::warn!("Pre-write pull failed (continuing anyway): {e}");
        }

        // Write file
        let full = self.repo_path.join(rel_path);
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&full, content)?;

        // Stage + commit + push
        add_commit_push(&repo, &[rel_path], commit_msg)?;

        Ok(())
    }

    /// Delete a file, commit, and push to origin.
    pub fn delete_and_sync(&self, rel_path: &str, commit_msg: &str) -> anyhow::Result<()> {
        let repo = self
            .repo
            .lock()
            .map_err(|e| anyhow::anyhow!("mutex poisoned: {e}"))?;

        if let Err(e) = pull_impl(&repo) {
            tracing::warn!("Pre-delete pull failed (continuing anyway): {e}");
        }

        let full = self.repo_path.join(rel_path);
        if full.exists() {
            std::fs::remove_file(&full)?;
        }

        // Remove from index
        let mut index = repo.index()?;
        let _ = index.remove_path(Path::new(rel_path));
        index.write()?;

        let tree_oid = index.write_tree()?;
        let tree = repo.find_tree(tree_oid)?;
        let sig = default_signature(&repo)?;

        let parent = head_commit(&repo);
        let parents: Vec<&git2::Commit<'_>> = parent.iter().collect();
        repo.commit(Some("HEAD"), &sig, &sig, commit_msg, &tree, &parents)?;

        push_to_origin(&repo)?;

        Ok(())
    }

    /// Write a JSON-serializable value as a file, commit, and push.
    pub fn write_json_and_sync<T: serde::Serialize + ?Sized>(
        &self,
        rel_path: &str,
        value: &T,
        commit_msg: &str,
    ) -> anyhow::Result<()> {
        let content = serde_json::to_string_pretty(value)?;
        self.write_and_sync(rel_path, &content, commit_msg)
    }
}

/// Bridge `GitStorage` into the `nexo_notes::NoteStorage` trait.

impl nexo_notes::NoteStorage for GitStorage {
    fn write_note(&self, filename: &str, content: &str) -> anyhow::Result<()> {
        self.write_and_sync(
            &format!("NOTES/{filename}"),
            content,
            &format!("Add note: {filename}"),
        )
    }

    fn read_note(&self, filename: &str) -> anyhow::Result<String> {
        self.read_file(&format!("NOTES/{filename}"))
    }

    fn list_notes(&self) -> anyhow::Result<Vec<String>> {
        Ok(self
            .list_files("NOTES/")?
            .into_iter()
            .filter(|f| f != "SUMMARY.md")
            .collect())
    }

    fn delete_note(&self, filename: &str) -> anyhow::Result<bool> {
        let path = format!("NOTES/{filename}");
        if !self.file_exists(&path) {
            return Ok(false);
        }
        self.delete_and_sync(&path, &format!("Remove note: {filename}"))?;
        Ok(true)
    }

    fn write_summary(&self, content: &str) -> anyhow::Result<()> {
        self.write_and_sync("NOTES/SUMMARY.md", content, "Update notes summary")
    }

    fn read_summary(&self) -> anyhow::Result<Option<String>> {
        if !self.file_exists("NOTES/SUMMARY.md") {
            return Ok(None);
        }
        Ok(Some(self.read_file("NOTES/SUMMARY.md")?))
    }
}

/// Pull remote changes into the repository when a fast-forward update is available.

fn pull_impl(repo: &Repository) -> anyhow::Result<()> {
    // Check if remote exists
    let mut remote = match repo.find_remote("origin") {
        Ok(r) => r,
        Err(_) => return Ok(()), // no remote configured
    };

    let mut fo = FetchOptions::new();
    fo.remote_callbacks(ssh_callbacks());
    remote.fetch(&["main"], Some(&mut fo), None)?;
    drop(remote);

    // Try to fast-forward to FETCH_HEAD
    let fetch_head = match repo.find_reference("refs/remotes/origin/main") {
        Ok(r) => r,
        Err(_) => return Ok(()), // remote has no main branch yet
    };

    let fetch_commit = repo.reference_to_annotated_commit(&fetch_head)?;

    // Check if we have a HEAD at all
    match repo.head() {
        Ok(head) => {
            let head_commit = repo.reference_to_annotated_commit(&head)?;
            let (analysis, _pref) = repo.merge_analysis(&[&fetch_commit])?;

            if analysis.is_up_to_date() {
                return Ok(());
            }

            if analysis.is_fast_forward() {
                let target_oid = fetch_commit.id();
                let target = repo.find_object(target_oid, Some(ObjectType::Commit))?;
                repo.checkout_tree(&target, None)?;

                let mut head_ref = repo.head()?;
                head_ref.set_target(target_oid, "fast-forward pull")?;
            } else {
                tracing::warn!(
                    "Cannot fast-forward from {} to {}; skipping pull",
                    head_commit.id(),
                    fetch_commit.id()
                );
            }
        }
        Err(_) => {
            // No HEAD yet (completely empty local repo). Set main to the fetched commit.
            let target_oid = fetch_commit.id();
            let target = repo.find_object(target_oid, Some(ObjectType::Commit))?;
            repo.checkout_tree(&target, None)?;
            repo.reference(
                "refs/heads/main",
                target_oid,
                true,
                "initial checkout from remote",
            )?;
            repo.set_head("refs/heads/main")?;
        }
    }

    Ok(())
}

fn ssh_callbacks() -> RemoteCallbacks<'static> {
    let mut cbs = RemoteCallbacks::new();
    cbs.credentials(|_url, username_from_url, _allowed_types| {
        let username = username_from_url.unwrap_or("git");
        // Try SSH agent first, then fall back to default key
        Cred::ssh_key_from_agent(username).or_else(|_| {
            let home = dirs::home_dir().unwrap_or_default();
            Cred::ssh_key(
                username,
                Some(&home.join(".ssh/id_ed25519.pub")),
                &home.join(".ssh/id_ed25519"),
                None,
            )
        })
    });
    cbs
}

fn default_signature(repo: &Repository) -> anyhow::Result<Signature<'_>> {
    repo.signature().or_else(|_| {
        Signature::now("nexo", "nexo@localhost").map_err(|e| anyhow::anyhow!("signature: {e}"))
    })
}

fn head_commit(repo: &Repository) -> Option<git2::Commit<'_>> {
    repo.head().ok().and_then(|h| h.peel_to_commit().ok())
}

fn add_commit_push(repo: &Repository, paths: &[&str], message: &str) -> anyhow::Result<()> {
    let mut index = repo.index()?;
    for path in paths {
        index.add_path(Path::new(path))?;
    }
    index.write()?;

    let tree_oid = index.write_tree()?;
    let tree = repo.find_tree(tree_oid)?;
    let sig = default_signature(repo)?;

    let parent = head_commit(repo);
    let parents: Vec<&git2::Commit<'_>> = parent.iter().collect();
    repo.commit(Some("HEAD"), &sig, &sig, message, &tree, &parents)?;

    push_to_origin(repo)?;

    Ok(())
}

fn push_to_origin(repo: &Repository) -> anyhow::Result<()> {
    let mut remote = match repo.find_remote("origin") {
        Ok(r) => r,
        Err(_) => {
            tracing::debug!("No 'origin' remote configured; skipping push");
            return Ok(());
        }
    };

    let mut po = PushOptions::new();
    po.remote_callbacks(ssh_callbacks());
    remote.push(&["refs/heads/main:refs/heads/main"], Some(&mut po))?;

    Ok(())
}
