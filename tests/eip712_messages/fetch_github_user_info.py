
import requests

def get_github_user_info(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    
    if response.status_code == 200:
        user_data = response.json()
        return {
            'name': user_data.get('name'),
            'public_repos': user_data.get('public_repos'),
            'followers': user_data.get('followers'),
            'following': user_data.get('following'),
            'created_at': user_data.get('created_at')
        }
    else:
        return None

if __name__ == "__main__":
    username = input("Enter GitHub username: ")
    info = get_github_user_info(username)
    
    if info:
        print(f"Name: {info['name']}")
        print(f"Public Repositories: {info['public_repos']}")
        print(f"Followers: {info['followers']}")
        print(f"Following: {info['following']}")
        print(f"Account Created: {info['created_at']}")
    else:
        print("User not found or API request failed.")