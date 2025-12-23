import requests
import sys

def get_user_repositories(username, page=1, per_page=30):
    url = f"https://api.github.com/users/{username}/repos"
    params = {
        'page': page,
        'per_page': per_page,
        'type': 'owner',
        'sort': 'updated',
        'direction': 'desc'
    }
    headers = {'Accept': 'application/vnd.github.v3+json'}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repositories: {e}", file=sys.stderr)
        return None

def display_repositories(repos):
    if not repos:
        print("No repositories found or error occurred.")
        return
    
    for repo in repos:
        name = repo.get('name', 'N/A')
        description = repo.get('description', 'No description')
        stars = repo.get('stargazers_count', 0)
        forks = repo.get('forks_count', 0)
        language = repo.get('language', 'Not specified')
        updated = repo.get('updated_at', 'N/A')[:10]
        
        print(f"Repository: {name}")
        print(f"  Description: {description}")
        print(f"  Language: {language}")
        print(f"  Stars: {stars} | Forks: {forks}")
        print(f"  Last updated: {updated}")
        print("-" * 50)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_github_repos.py <github_username> [page_number]")
        sys.exit(1)
    
    username = sys.argv[1]
    page = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    print(f"Fetching repositories for user: {username} (Page {page})")
    print("=" * 50)
    
    repos = get_user_repositories(username, page)
    display_repositories(repos)
    
    if repos and len(repos) == 30:
        print(f"\nNote: More repositories available. Try page {page + 1}.")

if __name__ == "__main__":
    main()