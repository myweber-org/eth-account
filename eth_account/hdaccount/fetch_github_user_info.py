import requests
import sys

def get_github_user_info(username):
    """Fetch and display public information for a given GitHub username."""
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        user_data = response.json()
        print(f"Username: {user_data.get('login')}")
        print(f"Name: {user_data.get('name', 'Not provided')}")
        print(f"Public Repos: {user_data.get('public_repos')}")
        print(f"Followers: {user_data.get('followers')}")
        print(f"Following: {user_data.get('following')}")
        print(f"Profile URL: {user_data.get('html_url')}")
    except requests.exceptions.HTTPError as e:
        print(f"Error: User '{username}' not found or API error.", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user_info.py <username>", file=sys.stderr)
        sys.exit(1)
    get_github_user_info(sys.argv[1])
import requests
import json

def get_github_user_info(username):
    """
    Fetch public information for a GitHub user.
    
    Args:
        username (str): GitHub username
    
    Returns:
        dict: User information or error message
    """
    url = f"https://api.github.com/users/{username}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        user_data = response.json()
        
        # Extract key information
        filtered_data = {
            'login': user_data.get('login'),
            'name': user_data.get('name'),
            'company': user_data.get('company'),
            'blog': user_data.get('blog'),
            'location': user_data.get('location'),
            'email': user_data.get('email'),
            'hireable': user_data.get('hireable'),
            'bio': user_data.get('bio'),
            'public_repos': user_data.get('public_repos'),
            'followers': user_data.get('followers'),
            'following': user_data.get('following'),
            'created_at': user_data.get('created_at'),
            'updated_at': user_data.get('updated_at')
        }
        
        return filtered_data
        
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            return {'error': f'User {username} not found'}
        return {'error': f'HTTP error occurred: {e}'}
    except requests.exceptions.RequestException as e:
        return {'error': f'Request failed: {e}'}

def save_user_info_to_file(username, user_info, filename=None):
    """
    Save user information to a JSON file.
    
    Args:
        username (str): GitHub username
        user_info (dict): User information dictionary
        filename (str, optional): Output filename
    """
    if filename is None:
        filename = f"{username}_github_info.json"
    
    with open(filename, 'w') as f:
        json.dump(user_info, f, indent=2)
    
    print(f"User information saved to {filename}")

def display_user_info(user_info):
    """
    Display user information in a readable format.
    
    Args:
        user_info (dict): User information dictionary
    """
    if 'error' in user_info:
        print(f"Error: {user_info['error']}")
        return
    
    print("\n" + "="*50)
    print("GitHub User Information")
    print("="*50)
    
    for key, value in user_info.items():
        if value is not None:
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("="*50)

def main():
    """
    Main function to demonstrate the script functionality.
    """
    # Example usage
    username = "octocat"  # GitHub's example user
    
    print(f"Fetching information for user: {username}")
    user_info = get_github_user_info(username)
    
    display_user_info(user_info)
    
    # Save to file
    save_user_info_to_file(username, user_info)
    
    # Return the data for potential further processing
    return user_info

if __name__ == "__main__":
    main()