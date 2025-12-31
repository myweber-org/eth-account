import requests

def fetch_contributors(repo_owner, repo_name):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contributors"
    response = requests.get(url)
    
    if response.status_code == 200:
        contributors = response.json()
        print(f"Contributors for {repo_owner}/{repo_name}:")
        for contributor in contributors:
            print(f"- {contributor['login']}: {contributor['contributions']} contributions")
    else:
        print(f"Failed to fetch contributors. Status code: {response.status_code}")

if __name__ == "__main__":
    fetch_contributors("torvalds", "linux")