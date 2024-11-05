import os.path

import config
from get_info import get_repo_path_from_url, get_repo_data, get_languages_data, get_contributors_data, get_license_data, \
    get_readme_content
from code_loader import code_loader
from llm_classification import analyze_with_agent
import time

def make_classification(repo_url, repo_name):
    # 从仓库URL中提取仓库路径
    repo_path = get_repo_path_from_url(repo_url)

    # 如果仓库路径无效，则打印错误信息并退出函数
    if not repo_path:
        print("Invalid GitHub URL")
        return

    # 使用GitHub令牌进行身份验证
    headers = {
        "Authorization": f"Bearer {config.GITHUB_TOKEN}"
    }

    # 获取仓库的基本信息
    repo_data = get_repo_data(repo_path, headers)
    # 如果返回的信息中包含message字段，则表示请求出错，打印错误信息并退出函数
    if 'message' in repo_data:
        print("Error:", repo_data['message'])
        return

    # 分别获取仓库的语言数据、贡献者数据和README内容
    languages_data = get_languages_data(repo_path, headers)
    contributors_data = get_contributors_data(repo_path, headers)
    readme_content = get_readme_content(repo_path, headers)

    # 如果README内容不可用，则打印警告信息
    if readme_content == "Not available":
        print("Warning：README Content Not available")

    # 使用获取到的数据进行分析，并得到分析任务的ID
    analysis_report, code_quality = analyze_with_agent(repo_name, repo_data, languages_data, len(contributors_data),
                                                       readme_content)

    license_data = get_license_data(repo_path, headers)
    print(f"以下是对仓库{repo_name}的多标签分类报告:")

    print("授权许可:", license_data.get('license', {}).get('name', 'Not available'))

    print("仓库标签:\n", analysis_report)

    print("按照6个分类标准对于代码质量评分如下（满分100分）:\n", code_quality)


if __name__ == "__main__":
    # repo_url = input("请输入仓库链接http: ")
    # repo_ssh = input("请输入仓库链接ssh: ")
    # repo_name = input("请输入仓库名称: ")

    repo_url = 'https://github.com/decisionintelligence/TFB'
    repo_ssh = 'git@github.com:decisionintelligence/TFB.git'
    repo_name = 'TFB'

    if not os.path.exists('cache/' + repo_name):
        code_loader(repo_ssh)


    make_classification(repo_url, repo_name)

