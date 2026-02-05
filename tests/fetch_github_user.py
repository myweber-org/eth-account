import requests

def fetch_github_user(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"User {username} not found"}

if __name__ == "__main__":
    user_data = fetch_github_user("octocat")
    print(user_data)
import requests
import json

def fetch_github_user(username):
    """Fetch public information for a given GitHub username."""
    url = f"https://api.github.com/users/{username}"
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        user_data = response.json()
        return user_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except json.JSONDecodeError:
        print("Error parsing JSON response.")
        return None

def display_user_info(user_data):
    """Display selected user information in a readable format."""
    if not user_data:
        print("No user data to display.")
        return

    print("\n--- GitHub User Profile ---")
    print(f"Username: {user_data.get('login', 'N/A')}")
    print(f"Name: {user_data.get('name', 'N/A')}")
    print(f"Bio: {user_data.get('bio', 'N/A')}")
    print(f"Public Repos: {user_data.get('public_repos', 'N/A')}")
    print(f"Followers: {user_data.get('followers', 'N/A')}")
    print(f"Following: {user_data.get('following', 'N/A')}")
    print(f"Profile URL: {user_data.get('html_url', 'N/A')}")
    print("---------------------------\n")

if __name__ == "__main__":
    target_username = input("Enter a GitHub username: ").strip()
    if target_username:
        data = fetch_github_user(target_username)
        display_user_info(data)
    else:
        print("No username provided.")