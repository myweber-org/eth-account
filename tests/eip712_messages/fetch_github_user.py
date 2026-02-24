import requests

def fetch_github_user(username):
    """
    Fetches public details and repositories of a GitHub user.
    """
    user_url = f"https://api.github.com/users/{username}"
    repos_url = f"https://api.github.com/users/{username}/repos"

    try:
        user_response = requests.get(user_url)
        user_response.raise_for_status()
        user_data = user_response.json()

        repos_response = requests.get(repos_url)
        repos_response.raise_for_status()
        repos_data = repos_response.json()

        print(f"User: {user_data.get('login')}")
        print(f"Name: {user_data.get('name')}")
        print(f"Public Repositories: {user_data.get('public_repos')}")
        print("\nRepository List:")
        for repo in repos_data:
            print(f"  - {repo.get('name')}: {repo.get('description')}")

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    fetch_github_user("octocat")