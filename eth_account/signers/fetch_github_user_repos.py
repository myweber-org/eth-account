import requests

def fetch_user_repositories(username, per_page=10, max_pages=5):
    base_url = "https://api.github.com/users/{}/repos"
    url = base_url.format(username)
    repos = []
    
    for page in range(1, max_pages + 1):
        params = {
            'page': page,
            'per_page': per_page,
            'sort': 'updated',
            'direction': 'desc'
        }
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"Error fetching data: {response.status_code}")
            break
        
        page_repos = response.json()
        if not page_repos:
            break
        
        repos.extend(page_repos)
        print(f"Fetched page {page}: {len(page_repos)} repositories")
    
    return repos

def display_repository_info(repos):
    if not repos:
        print("No repositories found.")
        return
    
    print(f"\nTotal repositories fetched: {len(repos)}")
    print("=" * 60)
    
    for idx, repo in enumerate(repos, 1):
        print(f"{idx}. {repo['name']}")
        print(f"   Description: {repo['description'] or 'No description'}")
        print(f"   Language: {repo['language'] or 'Not specified'}")
        print(f"   Stars: {repo['stargazers_count']}")
        print(f"   Forks: {repo['forks_count']}")
        print(f"   Updated: {repo['updated_at'][:10]}")
        print(f"   URL: {repo['html_url']}")
        print("-" * 40)

def main():
    username = input("Enter GitHub username: ").strip()
    if not username:
        print("Username cannot be empty.")
        return
    
    print(f"\nFetching repositories for user: {username}")
    repos = fetch_user_repositories(username)
    display_repository_info(repos)

if __name__ == "__main__":
    main()