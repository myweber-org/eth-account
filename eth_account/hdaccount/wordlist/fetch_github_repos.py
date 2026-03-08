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
    
    fetch_repositories(username, sort_by, direction)import requests
import sys

def fetch_repositories(username, page=1, per_page=30):
    url = f"https://api.github.com/users/{username}/repos"
    params = {"page": page, "per_page": per_page}
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: Unable to fetch repositories (Status code: {response.status_code})")
        return None

def display_repositories(repos):
    if not repos:
        print("No repositories found.")
        return
    
    for repo in repos:
        print(f"Name: {repo['name']}")
        print(f"Description: {repo['description'] or 'No description'}")
        print(f"URL: {repo['html_url']}")
        print(f"Stars: {repo['stargazers_count']}")
        print(f"Forks: {repo['forks_count']}")
        print("-" * 50)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_github_repos.py <username> [page] [per_page]")
        sys.exit(1)
    
    username = sys.argv[1]
    page = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    per_page = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    
    repos = fetch_repositories(username, page, per_page)
    if repos:
        display_repositories(repos)

if __name__ == "__main__":
    main()