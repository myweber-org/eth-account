import requests
import sys

def fetch_issues(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    params = {'state': 'open'}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch issues: {response.status_code}")
        return []

def display_issues(issues):
    if not issues:
        print("No open issues found.")
        return
    for issue in issues:
        print(f"#{issue['number']}: {issue['title']}")
        print(f"    URL: {issue['html_url']}")
        print(f"    Created by: {issue['user']['login']}")
        print()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_github_issues.py <owner> <repo>")
        sys.exit(1)
    owner = sys.argv[1]
    repo = sys.argv[2]
    issues = fetch_issues(owner, repo)
    display_issues(issues)