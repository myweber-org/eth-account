
import requests

def get_github_user(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

if __name__ == "__main__":
    user_data = get_github_user("octocat")
    if user_data:
        print(f"Name: {user_data.get('name')}")
        print(f"Bio: {user_data.get('bio')}")
        print(f"Public Repos: {user_data.get('public_repos')}")
    else:
        print("User not found or API error.")import requests

def get_github_user_info(username):
    """Fetch public information for a given GitHub username."""
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    
    if response.status_code == 200:
        user_data = response.json()
        return {
            'login': user_data.get('login'),
            'name': user_data.get('name'),
            'public_repos': user_data.get('public_repos'),
            'followers': user_data.get('followers'),
            'following': user_data.get('following'),
            'html_url': user_data.get('html_url')
        }
    else:
        return None

def main():
    username = input("Enter a GitHub username: ")
    user_info = get_github_user_info(username)
    
    if user_info:
        print(f"\nGitHub User: {user_info['login']}")
        print(f"Name: {user_info['name']}")
        print(f"Public Repositories: {user_info['public_repos']}")
        print(f"Followers: {user_info['followers']}")
        print(f"Following: {user_info['following']}")
        print(f"Profile URL: {user_info['html_url']}")
    else:
        print(f"User '{username}' not found or an error occurred.")

if __name__ == "__main__":
    main()