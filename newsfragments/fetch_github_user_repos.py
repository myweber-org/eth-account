import requests
import sys

def fetch_user_repos(username, per_page=30):
    base_url = "https://api.github.com/users/{}/repos"
    page = 1
    repos = []

    while True:
        url = base_url.format(username)
        params = {'page': page, 'per_page': per_page}
        response = requests.get(url, params=params)

        if response.status_code != 200:
            print(f"Error: Unable to fetch repositories (Status: {response.status_code})")
            sys.exit(1)

        data = response.json()
        if not data:
            break

        repos.extend(data)
        page += 1

    return repos

def display_repos(repos):
    if not repos:
        print("No repositories found.")
        return

    print(f"Total repositories: {len(repos)}")
    for repo in repos:
        print(f"- {repo['name']}: {repo['description'] or 'No description'}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetch_github_user_repos.py <github_username>")
        sys.exit(1)

    username = sys.argv[1]
    repositories = fetch_user_repos(username)
    display_repos(repositories)