import requests
import sys

def get_user_repositories(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    if response.status_code == 200:
        repos = response.json()
        return [repo['name'] for repo in repos]
    else:
        print(f"Error: Unable to fetch repositories (Status code: {response.status_code})")
        return []

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user_repos.py <username>")
        sys.exit(1)
    
    username = sys.argv[1]
    repos = get_user_repositories(username)
    
    if repos:
        print(f"Repositories for user '{username}':")
        for repo in repos:
            print(f"  - {repo}")
    else:
        print(f"No repositories found for user '{username}'.")import requests

def fetch_github_repos(username, per_page=30, page=1):
    url = f"https://api.github.com/users/{username}/repos"
    params = {
        "per_page": per_page,
        "page": page
    }
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        repos = response.json()
        
        if not repos:
            print(f"No repositories found for user '{username}'.")
            return []
        
        print(f"Repositories for {username} (Page {page}):")
        for repo in repos:
            print(f"- {repo['name']}: {repo['description'] or 'No description'}")
        
        return repos
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        if response.status_code == 404:
            print(f"User '{username}' not found.")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    
    return []

if __name__ == "__main__":
    username = input("Enter GitHub username: ").strip()
    if username:
        fetch_github_repos(username)