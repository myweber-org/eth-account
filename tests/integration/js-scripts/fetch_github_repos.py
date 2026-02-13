import requests

def fetch_github_repos(username):
    """Fetch public repositories for a given GitHub username."""
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    if response.status_code == 200:
        repos = response.json()
        repo_list = [repo['name'] for repo in repos]
        return repo_list
    else:
        print(f"Failed to fetch repositories. Status code: {response.status_code}")
        return []

def main():
    username = input("Enter a GitHub username: ")
    repos = fetch_github_repos(username)
    if repos:
        print(f"Public repositories for {username}:")
        for repo in repos:
            print(f" - {repo}")
    else:
        print("No repositories found or an error occurred.")

if __name__ == "__main__":
    main()import requests

def fetch_github_repos(username, per_page=10, page=1):
    url = f"https://api.github.com/users/{username}/repos"
    params = {
        'per_page': per_page,
        'page': page
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        repos = response.json()
        for repo in repos:
            print(f"Name: {repo['name']}")
            print(f"Description: {repo['description']}")
            print(f"URL: {repo['html_url']}")
            print(f"Stars: {repo['stargazers_count']}")
            print("-" * 40)
        return repos
    else:
        print(f"Error: {response.status_code}")
        return None

if __name__ == "__main__":
    user = input("Enter GitHub username: ")
    fetch_github_repos(user)