"""Push a local commit to GitHub through the Git Data API.

Why this exists: from this network, `git push` to github.com:443 is
unavailable -- the HTTP proxy answers `CONNECT tunnel failed, response 502`
and a direct connection times out on 443 -- while api.github.com is reachable.
The Git Data API lets the same objects travel over the reachable host.

The script reconstructs, over the API, exactly the commit that git has already
built locally:

    blobs  -> tree (on top of the remote base tree) -> commit -> ref update

It is deliberately *not* a general-purpose push replacement: it is
single-shot, assumes the remote ref is the local commit's parent, and refuses
to run otherwise.  The token is taken from the GITHUB_TOKEN environment
variable and never written to disk.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

API = "https://api.github.com"
REPO = "mmcqinyong-github/Neuro-STOG"
BRANCH = "main"


def sh(*args: str) -> str:
    return subprocess.run(args, cwd=REPO_DIR, capture_output=True, text=True,
                          check=True).stdout.strip()


REPO_DIR = os.environ.get(
    "REPO_DIR", os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "..", "..", "Neuro-STOG")))


def api(url: str, token: str, method: str = "GET", payload: dict | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if e.code in (502, 503, 504) and attempt < 3:
                continue
            raise SystemExit(f"HTTP {e.code} on {method} {url}\n{detail}")
        except Exception as e:  # noqa: BLE001 -- transient transport
            if attempt < 3:
                continue
            raise SystemExit(f"transport error on {method} {url}: {e}")
    raise SystemExit(f"gave up on {method} {url}")


def main() -> None:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("set GITHUB_TOKEN")
    if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
        raise SystemExit(f"not a git repo: {REPO_DIR}")

    head = sh("git", "rev-parse", "HEAD")
    parent = sh("git", "rev-parse", "HEAD^")
    message = sh("git", "log", "-1", "--pretty=%B")
    author_name = sh("git", "log", "-1", "--pretty=%an")
    author_email = sh("git", "log", "-1", "--pretty=%ae")

    ref = api(f"{API}/repos/{REPO}/git/ref/heads/{BRANCH}", token)
    remote_sha = ref["object"]["sha"]
    force = False
    if remote_sha != parent:
        # The only case we tolerate is a previous run of this very script having
        # already landed a commit: its committer is us, so the ref is ours to
        # replace with a corrected commit built on the same parent.  Anything
        # else means real divergence -- stop.
        rc = api(f"{API}/repos/{REPO}/git/commits/{remote_sha}", token)
        ours = rc.get("committer", {}).get("email") == sh(
            "git", "log", "-1", "--pretty=%ce")
        if ours:
            print(f"replacing our own earlier API commit {remote_sha[:8]} "
                  f"(its parent chain is dropped; the new commit is built on "
                  f"{parent[:8]})")
            force = True
        else:
            raise SystemExit(
                f"refuse to push: remote {BRANCH} is {remote_sha[:8]} but the "
                f"local commit's parent is {parent[:8]}.  Reconcile with git first.")

    changed = sh("git", "diff", "--name-only", parent, head).splitlines()
    print(f"local {head[:8]}  parent/remote {remote_sha[:8]}  files {len(changed)}")

    tree_items = []
    for i, path in enumerate(changed, 1):
        # Blob content comes from the *git object store*, not the working tree:
        # this repository runs with core.autocrlf=input, so the working tree has
        # CRLF while the stored blob is LF.  Uploading working-tree bytes would
        # produce a different tree (and therefore a different commit id) than
        # the commit git built locally.
        try:
            content_bytes = subprocess.run(
                ["git", "cat-file", "blob", f"{head}:{path}"], cwd=REPO_DIR,
                capture_output=True, check=True).stdout
        except subprocess.CalledProcessError:
            tree_items.append(dict(path=path.replace("\\", "/"), mode="100644",
                                   type="blob", sha=None))          # deletion
            print(f"  [{i:>2}/{len(changed)}] delete {path}")
            continue
        content = base64.b64encode(content_bytes).decode()
        blob = api(f"{API}/repos/{REPO}/git/blobs", token, "POST",
                   dict(content=content, encoding="base64"))
        # Mode must be read from git, not from the filesystem: on Windows
        # os.access(path, os.X_OK) is true for ordinary files, so probing the
        # working tree would silently mark everything executable and make the
        # reconstructed tree differ from the local one even when every blob
        # matches byte for byte.
        mode = sh("git", "ls-tree", head, "--", path).split()[0]
        tree_items.append(dict(path=path.replace("\\", "/"), mode=mode,
                               type="blob", sha=blob["sha"]))
        print(f"  [{i:>2}/{len(changed)}] blob {path}")

    base_commit = api(f"{API}/repos/{REPO}/git/commits/{remote_sha}", token)
    tree = api(f"{API}/repos/{REPO}/git/trees", token, "POST",
               dict(base_tree=base_commit["tree"]["sha"], tree=tree_items))

    commit = api(f"{API}/repos/{REPO}/git/commits", token, "POST",
                 dict(message=message, tree=tree["sha"], parents=[parent],
                      author=dict(name=author_name, email=author_email,
                                  date=sh("git", "log", "-1", "--pretty=%aI")),
                      committer=dict(name=author_name, email=author_email,
                                     date=sh("git", "log", "-1", "--pretty=%cI"))))

    api(f"{API}/repos/{REPO}/git/refs/heads/{BRANCH}", token, "PATCH",
        dict(sha=commit["sha"], force=force))

    final = api(f"{API}/repos/{REPO}/git/ref/heads/{BRANCH}", token)
    local_tree = sh("git", "rev-parse", "HEAD^{tree}")
    remote_commit = api(f"{API}/repos/{REPO}/git/commits/{final['object']['sha']}",
                        token)
    print(f"\nremote {BRANCH} -> {final['object']['sha'][:8]} "
          f"({'MATCHES local HEAD' if final['object']['sha'] == head else 'differs from local HEAD'})")
    print(f"tree {remote_commit['tree']['sha'][:8]} vs local {local_tree[:8]} "
          f"({'IDENTICAL content' if remote_commit['tree']['sha'] == local_tree else 'CONTENT DIFFERS'})")


if __name__ == "__main__":
    sys.exit(main())
