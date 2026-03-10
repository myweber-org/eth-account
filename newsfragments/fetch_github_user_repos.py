import requests
import sys

def fetch_repositories(username, page=1, per_page=30):
    """Fetch repositories for a given GitHub username."""
    url = f"https://api.github.com/users/{username}/repos"
    params = {'page': page, 'per_page': per_page}
    headers = {'Accept': 'application/vnd.github.v3+json'}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}", file=sys.stderr)
        return None

def display_repositories(repos):
    """Display repository information."""
    if not repos:
        print("No repositories found.")
        return
    
    for repo in repos:
        print(f"Name: {repo.get('name', 'N/A')}")
        print(f"  Description: {repo.get('description', 'No description')}")
        print(f"  URL: {repo.get('html_url', 'N/A')}")
        print(f"  Stars: {repo.get('stargazers_count', 0)}")
        print(f"  Forks: {repo.get('forks_count', 0)}")
        print("-" * 50)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_github_user_repos.py <username> [page] [per_page]")
        sys.exit(1)
    
    username = sys.argv[1]
    page = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    per_page = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    
    repos = fetch_repositories(username, page, per_page)
    if repos is not None:
        display_repositories(repos)

if __name__ == "__main__":
    main()