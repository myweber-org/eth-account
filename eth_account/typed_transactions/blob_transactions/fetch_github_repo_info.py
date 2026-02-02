import requests
import json
import sys

def fetch_repo_info(owner, repo):
    """
    Fetch basic information about a GitHub repository.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {
        'Accept': 'application/vnd.github.v3+json',
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return {
            'name': data.get('name'),
            'full_name': data.get('full_name'),
            'description': data.get('description'),
            'html_url': data.get('html_url'),
            'stargazers_count': data.get('stargazers_count'),
            'forks_count': data.get('forks_count'),
            'open_issues_count': data.get('open_issues_count'),
            'language': data.get('language'),
            'created_at': data.get('created_at'),
            'updated_at': data.get('updated_at')
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repository info: {e}", file=sys.stderr)
        return None

def display_repo_info(info):
    """
    Display the fetched repository information in a formatted way.
    """
    if not info:
        print("No information to display.")
        return

    print("\n" + "="*50)
    print(f"Repository: {info['full_name']}")
    print("="*50)
    print(f"Description: {info['description'] or 'No description'}")
    print(f"URL: {info['html_url']}")
    print(f"Stars: {info['stargazers_count']:,}")
    print(f"Forks: {info['forks_count']:,}")
    print(f"Open Issues: {info['open_issues_count']:,}")
    print(f"Primary Language: {info['language'] or 'Not specified'}")
    print(f"Created: {info['created_at']}")
    print(f"Last Updated: {info['updated_at']}")
    print("="*50 + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_github_repo_info.py <owner> <repo>")
        print("Example: python fetch_github_repo_info.py octocat Hello-World")
        sys.exit(1)

    owner = sys.argv[1]
    repo = sys.argv[2]

    print(f"Fetching information for repository: {owner}/{repo}")
    repo_info = fetch_repo_info(owner, repo)

    if repo_info:
        display_repo_info(repo_info)
    else:
        print("Failed to fetch repository information.")
        sys.exit(1)