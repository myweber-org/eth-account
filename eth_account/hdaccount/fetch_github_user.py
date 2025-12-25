import requests

def fetch_github_user(username):
    """Fetch public information for a given GitHub username."""
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        user_data = response.json()
        return {
            'login': user_data.get('login'),
            'name': user_data.get('name'),
            'public_repos': user_data.get('public_repos'),
            'followers': user_data.get('followers'),
            'following': user_data.get('following'),
            'html_url': user_data.get('html_url')
        }
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None
    except Exception as err:
        print(f"An error occurred: {err}")
        return None

if __name__ == "__main__":
    username = input("Enter a GitHub username: ").strip()
    if username:
        user_info = fetch_github_user(username)
        if user_info:
            print(f"\nGitHub User: {user_info['login']}")
            print(f"Name: {user_info['name']}")
            print(f"Public Repositories: {user_info['public_repos']}")
            print(f"Followers: {user_info['followers']}")
            print(f"Following: {user_info['following']}")
            print(f"Profile URL: {user_info['html_url']}")
        else:
            print("Failed to fetch user information.")
    else:
        print("No username provided.")
import requests
import sys

def get_github_user(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: Unable to fetch user data. Status code: {response.status_code}")
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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user.py <username>")
        sys.exit(1)
    
    username = sys.argv[1]
    user_data = get_github_user(username)
    display_user_info(user_data)