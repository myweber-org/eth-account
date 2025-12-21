import requests
import sys

def fetch_repositories(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    if response.status_code == 200:
        repos = response.json()
        return [repo['name'] for repo in repos]
    else:
        print(f"Error: Unable to fetch repositories (Status code: {response.status_code})")
        return []

def main():
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_repos.py <github_username>")
        sys.exit(1)
    
    username = sys.argv[1]
    repos = fetch_repositories(username)
    
    if repos:
        print(f"Repositories for {username}:")
        for repo in repos:
            print(f" - {repo}")
    else:
        print(f"No repositories found for {username}")

if __name__ == "__main__":
    main()