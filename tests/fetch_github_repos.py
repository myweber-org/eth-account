
import requests

def fetch_github_repos(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    if response.status_code == 200:
        repos = response.json()
        for repo in repos:
            print(f"Name: {repo['name']}")
            print(f"Description: {repo['description']}")
            print(f"URL: {repo['html_url']}")
            print("-" * 40)
    else:
        print(f"Failed to fetch repositories. Status code: {response.status_code}")

if __name__ == "__main__":
    user = input("Enter GitHub username: ")
    fetch_github_repos(user)
import requests
import sys

def fetch_github_repos(username, token=None, per_page=30, page=1):
    """
    Fetch repositories for a given GitHub username with pagination.
    Returns a list of repository names or None if request fails.
    """
    url = f"https://api.github.com/users/{username}/repos"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    
    params = {"per_page": per_page, "page": page}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        repos = response.json()
        return [repo["name"] for repo in repos]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repositories: {e}", file=sys.stderr)
        return None

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
        if repos:
            print(f"Repositories for {username} (page {page}, {per_page} per page):")
            for idx, repo in enumerate(repos, 1):
                print(f"{idx}. {repo}")
        else:
            print(f"No repositories found for {username} on page {page}.")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()