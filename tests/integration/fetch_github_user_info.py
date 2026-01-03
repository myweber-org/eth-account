
import requests
import json

def get_github_user_info(username):
    """Fetch public information for a given GitHub username."""
    url = f"https://api.github.com/users/{username}"
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'Python-Script'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        user_data = response.json()

        info = {
            'login': user_data.get('login'),
            'name': user_data.get('name'),
            'public_repos': user_data.get('public_repos'),
            'followers': user_data.get('followers'),
            'following': user_data.get('following'),
            'created_at': user_data.get('created_at'),
            'updated_at': user_data.get('updated_at')
        }
        return info

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
        return None
    except json.JSONDecodeError as json_err:
        print(f"JSON decode error: {json_err}")
        return None

def display_user_info(user_info):
    """Display the fetched user information in a readable format."""
    if user_info:
        print("GitHub User Information:")
        print("-" * 30)
        for key, value in user_info.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
    else:
        print("No user information to display.")

if __name__ == "__main__":
    username = input("Enter a GitHub username: ").strip()
    if username:
        user_info = get_github_user_info(username)
        display_user_info(user_info)
    else:
        print("No username provided.")