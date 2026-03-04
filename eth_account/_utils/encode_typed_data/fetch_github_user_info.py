import requests
import sys

def get_github_user(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    
    if response.status_code == 200:
        user_data = response.json()
        print(f"Username: {user_data.get('login')}")
        print(f"Name: {user_data.get('name', 'Not provided')}")
        print(f"Public Repositories: {user_data.get('public_repos')}")
        print(f"Followers: {user_data.get('followers')}")
        print(f"Following: {user_data.get('following')}")
        print(f"Profile URL: {user_data.get('html_url')}")
    else:
        print(f"Error: User '{username}' not found or API request failed.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user_info.py <username>")
        sys.exit(1)
    
    username = sys.argv[1]
    get_github_user(username)import requests
import sys

def get_github_user(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return {
            'name': data.get('name'),
            'public_repos': data.get('public_repos'),
            'followers': data.get('followers'),
            'following': data.get('following'),
            'created_at': data.get('created_at')
        }
    else:
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user_info.py <username>")
        sys.exit(1)
    
    username = sys.argv[1]
    user_info = get_github_user(username)
    
    if user_info:
        print(f"GitHub User: {username}")
        print(f"Name: {user_info['name']}")
        print(f"Public Repositories: {user_info['public_repos']}")
        print(f"Followers: {user_info['followers']}")
        print(f"Following: {user_info['following']}")
        print(f"Account Created: {user_info['created_at']}")
    else:
        print(f"Failed to fetch information for user: {username}")