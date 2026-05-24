import json
import os

from climlab_stardust_extension.utils.file_handling import _get_latest_commit_hash


def load_config(config_path='config.json'):
    """Load project configuration from a JSON file.

    Returns a dict with keys: aerosols_token, aerosols_tables_dict,
    proj_name, aerosols_opt_tables_http.
    Returns None if the config file cannot be loaded.
    """
    try:
        with open(config_path, "r") as f:
            project_config = json.load(f)
    except Exception:
        print(f'Configuration file {config_path} failed to load!')
        return None

    token = project_config['github']['token']
    tables_dict = project_config["aerosols_table_dict"]
    proj_name = project_config["project_name"]
    repo_name = project_config['github'].get(
        'repository_name',
        project_config['github'].get('materials_repository_name', ''),
    )
    commit_hash = project_config['github'].get('tables_commit')
    if commit_hash is None:
        commit_hash = _get_latest_commit_hash(
            project_config['github']['organization_name'],
            repo_name,
            token, proj_name,
        )
    opt_tables_http = (
        f"https://raw.githubusercontent.com/"
        f"{project_config['github']['organization_name']}/"
        f"{repo_name}/"
        f"{commit_hash}"
    )
    return {
        'aerosols_token': token,
        'aerosols_tables_dict': tables_dict,
        'proj_name': proj_name,
        'aerosols_opt_tables_http': opt_tables_http,
    }
