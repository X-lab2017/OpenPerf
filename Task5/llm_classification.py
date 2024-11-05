import os
import random
import time
from pathlib import Path
import numpy as np
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatZhipuAI
from langchain.globals import set_debug
import config
from pydantic.v1 import validator, Field
from pydantic.v1 import BaseModel

set_debug(True)


class CodeQuality(BaseModel):
    readability: int = Field(default=70, description="请为这(些)代码的可读性打分（0-100分）：")
    maintainability: int = Field(default=70, description="请为这(些)代码的可维护性打分（0-100分）：")
    consistencency: int = Field(default=70, description="请为这(些)代码的一致性打分（0-100分）：")
    complexity: int = Field(default=70, description="请为这(些)代码的简洁性打分（0-100分）：")
    robustness: int = Field(default=70, description="请为这(些)代码的健壮性打分（0-100分）：")
    modualrity: int = Field(default=70, description="请为这(些)代码的模块化打分（0-100分）：")

    @validator('readability', 'maintainability', 'consistencency', 'complexity', 'robustness', 'modualrity')
    @classmethod
    def validate_score(cls, field):
        if not 0 <= field <= 100:
            raise ValueError("Score must be between 0 and 100")
        return field


class RepoAnalysis(BaseModel):
    application_domain: str = Field(default="机器学习", description="请根据如上对于本仓库的描述，给出该仓库的应用领域：")
    development_stage: str = Field(default="初创", description="请根据如上对于本仓库的描述，给出该仓库的开发阶段：")
    community_activity: str = Field(default="活跃", description="请根据如上对于本仓库的描述，给出该仓库的社区活跃度：")
    tech_stack: str = Field(default="Python", description="请根据如上对于本仓库的描述，给出该仓库的技术栈")


prompt_set = {
    "repo_analysis_prompt": """请根据以下GitHub仓库信息进行分析并用中文回答：
    - 仓库名称: {reponame}
    - 描述: {repodescription}
    - 语言: {languages}
    - 贡献者数量: {contributors_count}
    - Star数: {star}
    - Fork数: {fork}
    - README内容: {readme_content}
    
    {format_instructions}
    """,

    "code_quality_prompt": """{file_content} 
    请根据以下标准分析代码的质量并用中文回答：
    - 可读性：代码是否易于理解，命名是否清晰，格式是否规范。
    - 可维护性：代码是否容易修改和扩展，是否遵循了设计模式和原则。
    - 一致性：代码风格是否统一，是否遵循了团队或项目的编码标准。
    - 简洁性：代码是否简洁，避免冗余和复杂性。
    - 健壮性：代码是否能够优雅地处理错误和异常情况。
    - 模块化：代码是否按照功能划分模块，模块之间是否低耦合。
    
    {format_instructions}
    """,
}


def analyze_repo_with_agent(llm, repo_data, languages_data, contributors_count, readme_content):
    repo_analysis_parser = PydanticOutputParser(pydantic_object=RepoAnalysis)
    repo_analysis_prompt = PromptTemplate.from_template(template=prompt_set["repo_analysis_prompt"], partial_variables={
        "format_instructions": repo_analysis_parser.get_format_instructions()})
    repo_analysis_chain = repo_analysis_prompt | llm | repo_analysis_parser

    repo_analysis_report = repo_analysis_chain.invoke(
        {'reponame': repo_data.get('name'), 'repodescription': repo_data.get('description'),
         'languages': languages_data,
         'contributors_count': contributors_count, 'star': repo_data.get('stargazers_count'),
         'fork': repo_data.get('forks_count'), 'readme_content': readme_content})

    report = f'应用领域：{repo_analysis_report.application_domain}，开发阶段：{repo_analysis_report.development_stage}，社区活跃度：{repo_analysis_report.community_activity}，技术栈：{repo_analysis_report.tech_stack}'
    return report


def analyze_code_quality(repo_name):
    path = f'cache/{repo_name}'
    files = get_all_files(path)
    code_files = [file for file in files if
                  file.endswith(('.py', '.js', '.java', '.cpp', '.c', '.rb', '.go'))]
    length = len(code_files)

    max_length = min(length, 5)

    random.shuffle(code_files)
    code_files = code_files[:max_length]

    llm = ChatZhipuAI(model='glm-4', api_key=config.ZHIPUAI_API_KEY, temperature=0.5)
    code_quality_parser = PydanticOutputParser(pydantic_object=CodeQuality)
    code_quality_prompt = PromptTemplate.from_template(template=prompt_set["code_quality_prompt"], partial_variables={
        'format_instructions': code_quality_parser.get_format_instructions()})

    code_quality_chain = code_quality_prompt | llm | code_quality_parser

    file_content = [read_file_to_string(code_file) for code_file in code_files]

    evaluations = []
    cnt = 0
    for content in file_content:
        while (1):
            try:
                evaluation = code_quality_chain.invoke({'file_content': content})
                break
            except:
                cnt += 1
                if cnt > 5:
                    print("代码应该出错了，请检查API配置和VPN设置，强制退出中......")
                    exit(-1)
                print("出错了，正在重试.......")
                time.sleep(5)  # wait for 5 seconds before retrying

        cnt = 0
        evaluations.append(evaluation)

    readability, maintainability, consistencency, complexity, robustness, modualrity = [], [], [], [], [], []

    for evaluation in evaluations:
        readability.append(evaluation.readability)

        maintainability.append(evaluation.maintainability)

        consistencency.append(evaluation.consistencency)

        complexity.append(evaluation.complexity)

        robustness.append(evaluation.robustness)

        modualrity.append(evaluation.modualrity)

    code_quality = f'可读性: {np.mean(readability)}\n可维护性: {np.mean(maintainability)}\n一致性: {np.mean(consistencency)}\n简洁性: {np.mean(complexity)}\n鲁棒性: {np.mean(robustness)}\n模块化: {np.mean(modualrity)}'
    return code_quality


def analyze_with_agent(repo_name, repo_data, languages_data, contributors_count, readme_content):
    llm = ChatZhipuAI(model='glm-4', api_key=config.ZHIPUAI_API_KEY)
    cnt = 0
    while True:
        try:
            repo_analysis_report = analyze_repo_with_agent(llm, repo_data, languages_data, contributors_count,
                                                           readme_content)
            break
        except:
            cnt += 1
            if cnt > 5:
                print("代码应该出错了，请检查API配置和VPN设置，强制退出中......")
                exit(-1)
            print("仓库分析热点出错了，正在重试.......")
            time.sleep(5)  # wait for 5 seconds before retrying

    code_quality_score = analyze_code_quality(repo_name)

    return repo_analysis_report, code_quality_score


def get_all_files(directory):
    files_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            files_list.append(os.path.join(root, file))
    return files_list


def read_file_to_string(file_path):
    try:
        path = Path(file_path)
        content = path.read_text(encoding='utf-8')  # 使用read_text读取文本内容
        return content
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except IOError as e:
        print(f"An I/O error occurred: {e}")
