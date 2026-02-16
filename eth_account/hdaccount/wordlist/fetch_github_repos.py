import requests
import sys

def fetch_repositories(username, sort_by='full_name', direction='asc'):
    url = f"https://api.github.com/users/{username}/repos"
    params = {'sort': sort_by, 'direction': direction}
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        repos = response.json()
        
        if not repos:
            print(f"No repositories found for user '{username}'.")
            return
        
        print(f"Repositories for {username} (sorted by {sort_by}, {direction}):")
        for repo in repos:
            print(f"- {repo['name']}: {repo['description'] or 'No description'}")
            print(f"  Stars: {repo['stargazers_count']} | Forks: {repo['forks_count']}")
            print(f"  URL: {repo['html_url']}")
            print()
            
    except requests.exceptions.HTTPError as e:
        print(f"Error fetching repositories: {e}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetch_github_repos.py <username> [sort_by] [direction]")
        print("sort_by options: created, updated, pushed, full_name (default)")
        print("direction options: asc (default), desc")
        sys.exit(1)
    
    username = sys.argv[1]
    sort_by = sys.argv[2] if len(sys.argv) > 2 else 'full_name'
    direction = sys.argv[3] if len(sys.argv) > 3 else 'asc'
    
    fetch_repositories(username, sort_by, direction)