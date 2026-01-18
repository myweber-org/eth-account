import requests
import sys

def fetch_github_repos(username, page=1, per_page=10):
    url = f"https://api.github.com/users/{username}/repos"
    params = {"page": page, "per_page": per_page}
    headers = {"Accept": "application/vnd.github.v3+json"}

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        repos = response.json()
        return repos
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repositories: {e}", file=sys.stderr)
        return []

def display_repos(repos):
    if not repos:
        print("No repositories found.")
        return

    for repo in repos:
        name = repo.get("name", "N/A")
        description = repo.get("description", "No description")
        stars = repo.get("stargazers_count", 0)
        forks = repo.get("forks_count", 0)
        print(f"Name: {name}")
        print(f"  Description: {description}")
        print(f"  Stars: {stars}, Forks: {forks}")
        print("-" * 40)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_github_repos.py <username> [page] [per_page]")
        sys.exit(1)

    username = sys.argv[1]
    page = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    per_page = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    repos = fetch_github_repos(username, page, per_page)
    display_repos(repos)

if __name__ == "__main__":
    main()