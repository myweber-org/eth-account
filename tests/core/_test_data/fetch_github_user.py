
import requests

def get_github_user(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def display_user_info(user_data):
    if user_data:
        print(f"Username: {user_data.get('login')}")
        print(f"Name: {user_data.get('name')}")
        print(f"Public Repos: {user_data.get('public_repos')}")
        print(f"Followers: {user_data.get('followers')}")
        print(f"Following: {user_data.get('following')}")
        print(f"Profile URL: {user_data.get('html_url')}")
    else:
        print("User not found.")

if __name__ == "__main__":
    username = input("Enter GitHub username: ")
    user_info = get_github_user(username)
    display_user_info(user_info)import requests
import json

def fetch_github_user(username):
    """
    Fetch public profile information for a given GitHub username.
    """
    url = f"https://api.github.com/users/{username}"
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        user_data = response.json()
        return user_data
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
    except json.JSONDecodeError as json_err:
        print(f"JSON decode error: {json_err}")
    return None

def display_user_info(user_data):
    """
    Display selected user information in a readable format.
    """
    if not user_data:
        print("No user data to display.")
        return

    print(f"GitHub Profile: {user_data.get('html_url')}")
    print(f"Username: {user_data.get('login')}")
    print(f"Name: {user_data.get('name', 'Not provided')}")
    print(f"Public Repositories: {user_data.get('public_repos')}")
    print(f"Followers: {user_data.get('followers')}")
    print(f"Following: {user_data.get('following')}")
    print(f"Account Created: {user_data.get('created_at')}")

if __name__ == "__main__":
    username = input("Enter a GitHub username: ").strip()
    if username:
        data = fetch_github_user(username)
        display_user_info(data)
    else:
        print("No username provided.")