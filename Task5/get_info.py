import requests


# 获取GitHub仓库路径
def get_repo_path_from_url(url):
    parts = url.split('/')
    if len(parts) >= 5:
        return f"{parts[3]}/{parts[4]}"
    else:
        return None


# 获取GitHub仓库基本信息
def get_repo_data(repo_path, headers):
    api_url = f"https://api.github.com/repos/{repo_path}"
    return requests.get(api_url, headers=headers).json()


# 获取GitHub仓库语言信息
def get_languages_data(repo_path, headers):
    languages_url = f"https://api.github.com/repos/{repo_path}/languages"
    return requests.get(languages_url, headers=headers).json()


# 获取GitHub仓库贡献者信息
def get_contributors_data(repo_path, headers):
    contributors_url = f"https://api.github.com/repos/{repo_path}/contributors"
    return requests.get(contributors_url, headers=headers).json()


# 获取GitHub仓库授权许可信息
def get_license_data(repo_path, headers):
    license_url = f"https://api.github.com/repos/{repo_path}/license"
    return requests.get(license_url, headers=headers).json()


# 获取GitHub仓库README内容
def get_readme_content(repo_path, headers):
    readme_url = f"https://api.github.com/repos/{repo_path}/readme"
    readme_data = requests.get(readme_url, headers=headers).json()
    download_url = readme_data.get('download_url')
    if download_url:
        return requests.get(download_url).text
    else:
        return "Not available"


# 获取GitHub仓库的代码文件列表
def get_repo_tree(repo_path, headers):
    tree_url = f"https://api.github.com/repos/{repo_path}/git/trees/main?recursive=1"
    response = requests.get(tree_url, headers=headers)
    if response.status_code == 404:
        tree_url = f"https://api.github.com/repos/{repo_path}/git/trees/master?recursive=1"
        response = requests.get(tree_url, headers=headers)
    return response.json()


# 获取单个文件内容
def get_file_content(file_url, headers):
    response = requests.get(file_url, headers=headers)
    return response.text


# 获取GitHub特定仓库主要代码
def print_repo_code(repo_path, headers):
    repo_tree = get_repo_tree(repo_path, headers)

    if 'tree' not in repo_tree:
        print("Error: No tree data found in the repository response.")
        return

    # 筛选出主要代码文件（如 .py, .js 等）
    code_files = [file for file in repo_tree['tree'] if
                  file['path'].endswith(('.py', '.js', '.java', '.cpp', '.c', '.rb', '.go'))]

    for file in code_files:
        file_url = file['url']
        file_content = get_file_content(file_url, headers)
        print(f"File: {file['path']}\n")
        print(file_content)
        print("\n" + "=" * 80 + "\n")
