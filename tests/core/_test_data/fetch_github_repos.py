import requests
import sys

def fetch_github_repos(username, per_page=30, page=1):
    """
    Fetch repositories for a given GitHub username with pagination.
    Returns a list of repository names or an empty list on error.
    """
    url = f"https://api.github.com/users/{username}/repos"
    params = {"per_page": per_page, "page": page}
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        repos = response.json()
        return [repo["name"] for repo in repos]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repositories: {e}", file=sys.stderr)
        return []
    except (KeyError, ValueError) as e:
        print(f"Error parsing response: {e}", file=sys.stderr)
        return []

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_github_repos.py <github_username> [per_page] [page]")
        sys.exit(1)
    
    username = sys.argv[1]
    per_page = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    page = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    repos = fetch_github_repos(username, per_page, page)
    
    if repos:
        print(f"Repositories for user '{username}' (page {page}, {per_page} per page):")
        for idx, repo in enumerate(repos, start=1):
            print(f"{idx}. {repo}")
    else:
        print(f"No repositories found or error occurred for user '{username}'.")

if __name__ == "__main__":
    main()