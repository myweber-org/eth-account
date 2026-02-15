import requests
import sys

def fetch_contributors(repo_owner, repo_name):
    """
    Fetch the list of contributors for a given GitHub repository.
    Returns a list of contributor usernames and contributions count.
    """
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contributors"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code}")
        return []
    
    contributors = response.json()
    result = []
    for contributor in contributors:
        result.append({
            'login': contributor['login'],
            'contributions': contributor['contributions']
        })
    return result

def display_top_contributors(contributors, top_n=5):
    """
    Display the top N contributors based on contributions count.
    """
    if not contributors:
        print("No contributors found.")
        return
    
    sorted_contributors = sorted(contributors, key=lambda x: x['contributions'], reverse=True)
    top_contributors = sorted_contributors[:top_n]
    
    print(f"Top {len(top_contributors)} contributors:")
    for idx, contributor in enumerate(top_contributors, start=1):
        print(f"{idx}. {contributor['login']} - {contributor['contributions']} contributions")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_github_contributors.py <repo_owner> <repo_name>")
        sys.exit(1)
    
    repo_owner = sys.argv[1]
    repo_name = sys.argv[2]
    
    contributors = fetch_contributors(repo_owner, repo_name)
    display_top_contributors(contributors)