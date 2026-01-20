import requests
import sys

def fetch_github_repos(username):
    """
    Fetch public repositories for a given GitHub username.
    Returns a list of repository names or an empty list on error.
    """
    url = f"https://api.github.com/users/{username}/repos"
    try:
        response = requests.get(url)
        response.raise_for_status()
        repos_data = response.json()
        return [repo['name'] for repo in repos_data]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}", file=sys.stderr)
        return []
    except (KeyError, TypeError) as e:
        print(f"Error parsing response: {e}", file=sys.stderr)
        return []

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_repos.py <github_username>")
        sys.exit(1)
    
    username = sys.argv[1]
    repos = fetch_github_repos(username)
    
    if repos:
        print(f"Public repositories for user '{username}':")
        for repo in repos:
            print(f"  - {repo}")
    else:
        print(f"No public repositories found for user '{username}' or an error occurred.")