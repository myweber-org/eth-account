import requests
import json

def fetch_repo_info(username, repo_name):
    """
    Fetch basic information about a GitHub repository.
    """
    url = f"https://api.github.com/repos/{username}/{repo_name}"
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        repo_data = response.json()
        return {
            "name": repo_data.get("name"),
            "full_name": repo_data.get("full_name"),
            "description": repo_data.get("description"),
            "html_url": repo_data.get("html_url"),
            "stargazers_count": repo_data.get("stargazers_count"),
            "forks_count": repo_data.get("forks_count"),
            "open_issues_count": repo_data.get("open_issues_count"),
            "language": repo_data.get("language"),
            "updated_at": repo_data.get("updated_at")
        }
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None
    except Exception as err:
        print(f"An error occurred: {err}")
        return None

def main():
    username = "torvalds"
    repo_name = "linux"
    info = fetch_repo_info(username, repo_name)
    if info:
        print("Repository Information:")
        for key, value in info.items():
            print(f"{key}: {value}")
    else:
        print("Failed to fetch repository information.")

if __name__ == "__main__":
    main()