import requests
import json

def fetch_github_repos(username):
    """Fetch public repositories for a given GitHub username."""
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    
    if response.status_code == 200:
        repos = response.json()
        repo_list = []
        for repo in repos:
            repo_info = {
                'name': repo['name'],
                'description': repo['description'],
                'url': repo['html_url'],
                'stars': repo['stargazers_count']
            }
            repo_list.append(repo_info)
        return repo_list
    else:
        print(f"Failed to fetch repositories. Status code: {response.status_code}")
        return None

def display_repos(repos):
    """Display repository information in a formatted way."""
    if repos:
        for repo in repos:
            print(f"Name: {repo['name']}")
            print(f"Description: {repo['description']}")
            print(f"URL: {repo['url']}")
            print(f"Stars: {repo['stars']}")
            print("-" * 40)
    else:
        print("No repositories to display.")

if __name__ == "__main__":
    username = input("Enter GitHub username: ")
    repos = fetch_github_repos(username)
    display_repos(repos)