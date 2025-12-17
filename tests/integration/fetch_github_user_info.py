
import requests
import sys

def get_github_user_info(username):
    """
    Fetch public information for a GitHub user.
    """
    url = f"https://api.github.com/users/{username}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        sys.exit(1)
    except Exception as err:
        print(f"Other error occurred: {err}")
        sys.exit(1)

def display_user_info(user_data):
    """
    Display selected user information.
    """
    print(f"Username: {user_data.get('login')}")
    print(f"Name: {user_data.get('name', 'Not provided')}")
    print(f"Public Repos: {user_data.get('public_repos')}")
    print(f"Followers: {user_data.get('followers')}")
    print(f"Following: {user_data.get('following')}")
    print(f"Profile URL: {user_data.get('html_url')}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user_info.py <username>")
        sys.exit(1)
    
    username = sys.argv[1]
    user_data = get_github_user_info(username)
    display_user_info(user_data)