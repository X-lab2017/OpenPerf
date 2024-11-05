import subprocess
import os

def code_loader(repo_url):
    # 设置GitHub仓库的URL和本地文件夹路径
    local_folder = 'cache'

    # 确保目标文件夹存在
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)

    # 切换到目标文件夹
    os.chdir(local_folder)

    # 使用subprocess执行git clone命令
    try:
        subprocess.run(['git', 'clone', '-b', 'master', repo_url], check=True)
        print(f"代码已成功下载到 {local_folder}")
    except subprocess.CalledProcessError as e:
        print(f"下载失败: {e}")