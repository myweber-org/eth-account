import requests
import sys

def fetch_repositories(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    if response.status_code == 200:
        repos = response.json()
        for repo in repos:
            print(f"Name: {repo['name']}")
            print(f"Description: {repo['description']}")
            print(f"URL: {repo['html_url']}")
            print(f"Stars: {repo['stargazers_count']}")
            print("-" * 40)
    else:
        print(f"Error: Unable to fetch repositories for user '{username}'")
        print(f"Status Code: {response.status_code}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_repos.py <github_username>")
        sys.exit(1)
    username = sys.argv[1]
    fetch_repositories(username)import requests
import sys

def fetch_github_repos(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    
    if response.status_code == 200:
        repos = response.json()
        repo_names = [repo['name'] for repo in repos]
        return repo_names
    else:
        print(f"Error: Unable to fetch repositories for user '{username}'")
        return []

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_repos.py <github_username>")
        sys.exit(1)
    
    username = sys.argv[1]
    repos = fetch_github_repos(username)
    
    if repos:
        print(f"Repositories for user '{username}':")
        for repo in repos:
            print(f"  - {repo}")
    else:
        print(f"No repositories found for user '{username}'")