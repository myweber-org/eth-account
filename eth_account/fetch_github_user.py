import requests

def get_github_user(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"User {username} not found or API error"}

if __name__ == "__main__":
    user_data = get_github_user("octocat")
    print(user_data)