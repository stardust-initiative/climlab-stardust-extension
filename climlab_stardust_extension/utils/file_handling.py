"""Utility functions for file handling, version tracking, and git info."""

import pooch
import requests
import os
import subprocess
import json


def _get_latest_commit_hash(owner, repo, token, proj_name):
    """Fetch the latest commit SHA from a GitHub repo, with local caching.

    The SHA is written to ``<pooch-cache>/repo_<repo>_hash.txt`` so that
    subsequent offline invocations can still resolve it.

    Parameters
    ----------
    owner : str   GitHub organisation or user.
    repo  : str   Repository name.
    token : str   Personal-access token for the GitHub API.
    proj_name : str   Project name used for the pooch cache directory.

    Returns
    -------
    str   40-character hex SHA of the latest commit on ``main``.
    """
    local_folder = pooch.os_cache(proj_name)
    os.makedirs(local_folder, exist_ok=True)
    hash_cache_file = os.path.join(local_folder, f'repo_{repo}_hash.txt')

    try:
        api_url = f"https://api.github.com/repos/{owner}/{repo}/commits/main"
        headers = {"Authorization": f"token {token}"}
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        sha = response.json()['sha']
        with open(hash_cache_file, 'w') as f:
            f.write(sha)
    except Exception:
        print(
            f"{repo} - Warning: Cannot load hash of latest commit. "
            "Using cached file if available."
        )
        if not os.path.exists(hash_cache_file):
            raise RuntimeError(
                'There is no cached commit hash, and you are not online!'
            )
        with open(hash_cache_file, 'r') as f:
            sha = f.readline().strip()

    return sha


def load_repo_table(repo_url, file_path, token, **kwargs):
    """Download or load a cached data file from a GitHub repository.

    Parameters
    ----------
    repo_url : str
        Base URL of the raw GitHub content
        (e.g. https://raw.githubusercontent.com/org/repo/commit_hash)
    file_path : str
        Path to the file within the repository
    token : str
        GitHub personal access token for authentication
    **kwargs :
        local_nc_folder : str, optional
            Local directory for caching files
        proj_name : str, optional
            Project name for pooch cache directory

    Returns
    -------
    cache_file : str
        Path to the local cached file
    download_file : bool
        True if the file was freshly downloaded
    """
    file_name = file_path.split('/')[-1]

    # --- resolve cache directory -------------------------------------------
    if 'local_nc_folder' in kwargs:
        cache_dir = kwargs['local_nc_folder']
    elif 'proj_name' in kwargs:
        # Use commit-hash-keyed subdirectory for content-addressable caching
        commit_hash = repo_url.rstrip('/').split('/')[-1]
        cache_dir = os.path.join(
            pooch.os_cache(kwargs['proj_name']), commit_hash[:12],
        )
    else:
        cache_dir = os.getcwd()

    cache_file = os.path.join(cache_dir, file_name)
    already_cached = os.path.exists(cache_file)

    # --- download via pooch if not already cached --------------------------
    # With commit-hash-keyed directories the file is content-addressed:
    # if it exists at this path it is correct by definition.
    if not already_cached:
        url = f"{repo_url}/{file_path}"
        try:
            pooch.retrieve(
                url,
                known_hash=None,
                fname=file_name,
                path=cache_dir,
                downloader=pooch.HTTPDownloader(
                    headers={"Authorization": f"token {token}"},
                ),
            )
        except Exception as e:
            # Fall back to the flat cache directory (pre-commit-hash layout)
            if 'proj_name' in kwargs:
                flat_path = os.path.join(
                    pooch.os_cache(kwargs['proj_name']), file_name,
                )
                if os.path.exists(flat_path):
                    print(
                        f"{file_name} - Warning: Cannot download file ({e}). "
                        "Using cached file from flat cache."
                    )
                    return flat_path, False
            raise

    download_file = not already_cached
    return cache_file, download_file


def get_package_versions():
    """Get installed package versions as JSON string."""
    output = subprocess.run(
        ["pip", "list", "--format=json"],
        capture_output=True, text=True
    )
    return output.stdout


def run_git_command(command, repo_path="."):
    """Run a git command and return its output."""
    try:
        return subprocess.run(
            ["git", "-C", repo_path] + command,
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return None


def get_git_info(repo_path="."):
    """Get Git commit hash, dirty status, and uncommitted changes."""
    name = run_git_command(["rev-parse", "--show-toplevel"], repo_path)
    if name is None:
        name = ''
    else:
        name = name.split('/')[-1]
    if name == '':
        return None
    else:
        commit_hash = run_git_command(["rev-parse", "HEAD"], repo_path)
        is_dirty = run_git_command(["status", "--porcelain"], repo_path) != ""
        uncommitted_changes = (
            run_git_command(["diff"], repo_path) if is_dirty else ""
        )
        return {
            'name': name,
            'data': {
                "commit_hash": commit_hash if commit_hash else "",
                "is_dirty": is_dirty,
                "uncommitted_changes": uncommitted_changes
            }
        }


def attach_version_info(ds, do_packages=True, do_git=True, repo_path="."):
    """Attach version information to an xarray Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to attach version info to
    do_packages : bool
        Whether to include installed package versions
    do_git : bool
        Whether to include git repository information
    repo_path : str or list of str
        Path(s) to git repositories to inspect
    """
    version_info = {}
    if do_packages:
        version_info['packages'] = get_package_versions()
    if do_git:
        repo_path_list = (
            repo_path if isinstance(repo_path, list) else [repo_path]
        )
        version_info_dict = {}
        for p in repo_path_list:
            git_info_p = get_git_info(repo_path=p)
            if isinstance(git_info_p, dict):
                version_info_dict[git_info_p['name']] = git_info_p['data']
        if len(version_info_dict) > 0:
            version_info['git'] = version_info_dict
    ds.attrs["version_info"] = json.dumps(version_info)
    return ds


def version_info_to_lines(version_info_str):
    """Parse version_info attribute string into human-readable lines."""
    version_info = json.loads(version_info_str)
    packages_list = (
        json.loads(version_info['packages'])
        if 'packages' in version_info else []
    )
    git_info_dict = (
        version_info['git'] if 'git' in version_info else {}
    )
    lines = []
    for v in packages_list:
        lines.append(f"Package version - {v['name']}: {v['version']}")
    if len(git_info_dict) > 0:
        for name, info in git_info_dict.items():
            lines.append(f'Git repo - {name}:')
            lines.append(f"Commit: {info['commit_hash']}:")
            lines.append(f"Is dirty: {info['is_dirty']}:")
            for line in info['uncommitted_changes'].split('\n'):
                lines.append(line)
    return lines
