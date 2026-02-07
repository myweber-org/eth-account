
import requests
import argparse
import sys

def fetch_repositories(username, sort_by='full_name', direction='asc'):
    """
    Fetch repositories for a given GitHub username.
    """
    url = f"https://api.github.com/users/{username}/repos"
    params = {
        'sort': sort_by,
        'direction': direction,
        'per_page': 100
    }
    headers = {
        'Accept': 'application/vnd.github.v3+json'
    }

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        repos = response.json()
        return repos
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repositories: {e}", file=sys.stderr)
        return None

def display_repositories(repos, show_details=False):
    """
    Display repository information.
    """
    if not repos:
        print("No repositories found.")
        return

    for repo in repos:
        print(f"Name: {repo['name']}")
        print(f"  Description: {repo['description'] or 'No description'}")
        print(f"  URL: {repo['html_url']}")
        print(f"  Stars: {repo['stargazers_count']}")
        print(f"  Forks: {repo['forks_count']}")
        if show_details:
            print(f"  Language: {repo['language'] or 'Not specified'}")
            print(f"  Created: {repo['created_at']}")
            print(f"  Updated: {repo['updated_at']}")
        print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description='Fetch GitHub user repositories.')
    parser.add_argument('username', help='GitHub username')
    parser.add_argument('--sort', choices=['created', 'updated', 'pushed', 'full_name'],
                        default='full_name', help='Sort repositories by field')
    parser.add_argument('--direction', choices=['asc', 'desc'],
                        default='asc', help='Sort direction')
    parser.add_argument('--details', action='store_true',
                        help='Show detailed repository information')

    args = parser.parse_args()

    repos = fetch_repositories(args.username, args.sort, args.direction)
    if repos is not None:
        display_repositories(repos, args.details)

if __name__ == '__main__':
    main()import requests
import sys

def fetch_repositories(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    if response.status_code == 200:
        repos = response.json()
        for repo in repos:
            print(f"Name: {repo['name']}")
            print(f"Description: {repo['description'] or 'No description'}")
            print(f"URL: {repo['html_url']}")
            print(f"Stars: {repo['stargazers_count']}")
            print("-" * 40)
        return True
    else:
        print(f"Failed to fetch repositories. Status code: {response.status_code}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user_repos.py <username>")
        sys.exit(1)
    
    username = sys.argv[1]
    fetch_repositories(username)