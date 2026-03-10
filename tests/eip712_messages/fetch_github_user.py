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
import requests
import sys

def get_github_user(username):
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        user_data = response.json()
        return user_data
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None
    except Exception as err:
        print(f"An error occurred: {err}")
        return None

def display_user_info(user_data):
    if user_data:
        print(f"Username: {user_data.get('login')}")
        print(f"Name: {user_data.get('name')}")
        print(f"Bio: {user_data.get('bio')}")
        print(f"Public Repos: {user_data.get('public_repos')}")
        print(f"Followers: {user_data.get('followers')}")
        print(f"Following: {user_data.get('following')}")
        print(f"Profile URL: {user_data.get('html_url')}")
    else:
        print("No user data to display.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user.py <github_username>")
        sys.exit(1)
    
    username = sys.argv[1]
    user_info = get_github_user(username)
    display_user_info(user_info)