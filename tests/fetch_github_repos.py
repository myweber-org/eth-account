import requests
import sys

def fetch_github_repos(username, token=None, per_page=30, page=1):
    """
    Fetch repositories for a given GitHub username with pagination support.
    """
    url = f"https://api.github.com/users/{username}/repos"
    headers = {}
    if token:
        headers['Authorization'] = f'token {token}'

    params = {
        'per_page': per_page,
        'page': page
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        repos = response.json()
        return repos
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repositories: {e}", file=sys.stderr)
        return None

def display_repos(repos):
    """
    Display repository details.
    """
    if not repos:
        print("No repositories found.")
        return

    for repo in repos:
        print(f"Name: {repo['name']}")
        print(f"  Description: {repo['description'] or 'No description'}")
        print(f"  URL: {repo['html_url']}")
        print(f"  Stars: {repo['stargazers_count']}")
        print(f"  Forks: {repo['forks_count']}")
        print("-" * 40)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_github_repos.py <username> [token] [per_page] [page]")
        sys.exit(1)

    username = sys.argv[1]
    token = sys.argv[2] if len(sys.argv) > 2 else None
    per_page = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    page = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    repos = fetch_github_repos(username, token, per_page, page)
    if repos is not None:
        display_repos(repos)

if __name__ == "__main__":
    main()import requests
import sys

def fetch_repositories(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    if response.status_code == 200:
        repos = response.json()
        return repos
    else:
        print(f"Error: Unable to fetch repositories for user '{username}'")
        return None

def display_repositories(repos):
    if not repos:
        print("No repositories found.")
        return
    print(f"Found {len(repos)} repositories:")
    for repo in repos:
        print(f"- {repo['name']}: {repo['description'] or 'No description'}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_repos.py <github_username>")
        sys.exit(1)
    username = sys.argv[1]
    repos = fetch_repositories(username)
    if repos:
        display_repositories(repos)