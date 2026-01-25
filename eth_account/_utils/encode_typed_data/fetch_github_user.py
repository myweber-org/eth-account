
import requests

def get_github_user(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    
    if response.status_code == 200:
        user_data = response.json()
        return {
            'name': user_data.get('name'),
            'bio': user_data.get('bio'),
            'public_repos': user_data.get('public_repos'),
            'followers': user_data.get('followers'),
            'following': user_data.get('following')
        }
    else:
        return None

if __name__ == "__main__":
    username = input("Enter GitHub username: ")
    user_info = get_github_user(username)
    
    if user_info:
        print(f"Name: {user_info['name']}")
        print(f"Bio: {user_info['bio']}")
        print(f"Public Repositories: {user_info['public_repos']}")
        print(f"Followers: {user_info['followers']}")
        print(f"Following: {user_info['following']}")
    else:
        print("User not found or API request failed.")import requests

def fetch_github_user(username):
    """
    Fetches public details of a GitHub user.

    Args:
        username (str): The GitHub username.

    Returns:
        dict: A dictionary containing user details if successful.
        None: If the request fails or user is not found.
    """
    url = f"https://api.github.com/users/{username}"
    headers = {"Accept": "application/vnd.github.v3+json"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    user_data = fetch_github_user("octocat")
    if user_data:
        print(f"Username: {user_data.get('login')}")
        print(f"Name: {user_data.get('name')}")
        print(f"Public Repos: {user_data.get('public_repos')}")
        print(f"Followers: {user_data.get('followers')}")
    else:
        print("Failed to fetch user data.")