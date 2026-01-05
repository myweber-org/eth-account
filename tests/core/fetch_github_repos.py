import requests
import sys

def fetch_github_repos(username, token=None, per_page=30):
    """
    Fetch repositories for a given GitHub username with pagination support.
    Returns a list of repository names or None if an error occurs.
    """
    repos = []
    page = 1
    headers = {'Accept': 'application/vnd.github.v3+json'}
    if token:
        headers['Authorization'] = f'token {token}'

    while True:
        url = f'https://api.github.com/users/{username}/repos'
        params = {'page': page, 'per_page': per_page}
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            if not data:
                break
            for repo in data:
                repos.append(repo['name'])
            page += 1
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}", file=sys.stderr)
            return None
    return repos

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetch_github_repos.py <username> [token]")
        sys.exit(1)
    
    username = sys.argv[1]
    token = sys.argv[2] if len(sys.argv) > 2 else None
    repositories = fetch_github_repos(username, token)
    
    if repositories is not None:
        print(f"Found {len(repositories)} repositories for user '{username}':")
        for repo in repositories:
            print(f"  - {repo}")
    else:
        print("Failed to fetch repositories.")