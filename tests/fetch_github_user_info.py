import requests
import json

def get_github_user_info(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    
    if response.status_code == 200:
        user_data = response.json()
        print(f"Username: {user_data['login']}")
        print(f"Name: {user_data.get('name', 'N/A')}")
        print(f"Bio: {user_data.get('bio', 'N/A')}")
        print(f"Public Repos: {user_data['public_repos']}")
        print(f"Followers: {user_data['followers']}")
        print(f"Following: {user_data['following']}")
        print(f"Profile URL: {user_data['html_url']}")
    else:
        print(f"Error: Unable to fetch data for user '{username}' (Status code: {response.status_code})")

if __name__ == "__main__":
    username = input("Enter a GitHub username: ")
    get_github_user_info(username)