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
    main()