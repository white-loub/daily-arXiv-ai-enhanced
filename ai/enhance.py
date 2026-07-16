import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict

import dotenv
import argparse
from tqdm import tqdm

import langchain_core.exceptions
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
try:
    from .structure import Structure
except ImportError:
    from structure import Structure

if os.path.exists('.env'):
    dotenv.load_dotenv()
SCRIPT_DIR = Path(__file__).resolve().parent
template = (SCRIPT_DIR / "template.txt").read_text()
system = (SCRIPT_DIR / "system.txt").read_text()

QWEN_THINKING_MODELS = ("qwen3.5-", "qwen3.6-", "qwen3.7-")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of parallel workers")
    return parser.parse_args()

def process_single_item(chain, item: Dict, language: str) -> Dict:
    """处理单个数据项"""
    try:
        response: Structure = chain.invoke({
            "language": language,
            "content": item['summary']
        })
        item['AI'] = response.model_dump()
    except langchain_core.exceptions.OutputParserException as e:
        # 尝试从错误信息中提取 JSON 字符串并修复
        error_msg = str(e)
        if "Function Structure arguments:" in error_msg:
            json_str = ""
            try:
                # 提取 JSON 字符串
                json_str = error_msg.split("Function Structure arguments:", 1)[1].strip().split('are not valid JSON')[0].strip()
                # 预处理 LaTeX 数学符号 - 使用四个反斜杠来确保正确转义
                json_str = json_str.replace('\\', '\\\\')
                # 尝试解析修复后的 JSON
                fixed_data = json.loads(json_str)
                item['AI'] = Structure.model_validate(fixed_data).model_dump()
                return item
            except Exception as json_e:
                print(f"Failed to fix JSON for {item['id']}: {json_e} {json_str}", file=sys.stderr)

        # 解析失败时中止本批次，避免发布占位内容
        raise
    return item


def create_structured_llm(model_name: str):
    """创建结构化输出模型，并为需要的 Qwen 模型关闭思考模式。"""
    llm_kwargs = {"model": model_name}
    if model_name.lower().startswith(QWEN_THINKING_MODELS):
        llm_kwargs["extra_body"] = {"enable_thinking": False}
        print('Thinking mode disabled for structured output', file=sys.stderr)

    return ChatOpenAI(**llm_kwargs).with_structured_output(
        Structure,
        method="function_calling"
    )


def create_chain(model_name: str):
    """创建用于论文增强的提示链。"""
    llm = create_structured_llm(model_name)
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(template=template)
    ])
    return prompt_template | llm

def process_all_items(data: List[Dict], model_name: str, language: str, max_workers: int) -> List[Dict]:
    """并行处理所有数据项"""
    print('Connect to:', model_name, file=sys.stderr)
    if not data:
        return []

    chain = create_chain(model_name)

    # 先同步处理一条作为预检。模型配置或权限错误时只产生一次请求。
    # 使用线程池并行处理
    processed_data = [None] * len(data)  # 预分配结果列表
    try:
        processed_data[0] = process_single_item(chain, data[0], language)
    except Exception as e:
        item_id = data[0].get('id', 'unknown')
        raise RuntimeError(f"AI preflight failed for item {item_id}") from e

    if len(data) == 1:
        return processed_data

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 预检成功后再提交剩余任务
        future_to_idx = {
            executor.submit(process_single_item, chain, item, language): idx
            for idx, item in enumerate(data[1:], start=1)
        }

        # 使用tqdm显示进度
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(future_to_idx),
            desc="Processing items"
        ):
            idx = future_to_idx[future]
            try:
                result = future.result()
                processed_data[idx] = result
            except Exception as e:
                # 取消尚未开始的请求，避免同一配置错误继续消耗 API 配额。
                for pending_future in future_to_idx:
                    pending_future.cancel()
                item_id = data[idx].get('id', 'unknown')
                raise RuntimeError(f"AI enhancement failed for item {item_id}") from e

    return processed_data

def main():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", 'deepseek-chat')
    language = os.environ.get("LANGUAGE", 'Chinese')

    # 检查并删除目标文件
    target_file = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')
    if os.path.exists(target_file):
        os.remove(target_file)
        print(f'Removed existing file: {target_file}', file=sys.stderr)

    # 读取数据
    data = []
    with open(args.data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # 去重
    seen_ids = set()
    unique_data = []
    for item in data:
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_data.append(item)

    data = unique_data
    print('Open:', args.data, file=sys.stderr)
    
    # 并行处理所有数据
    processed_data = process_all_items(
        data,
        model_name,
        language,
        args.max_workers
    )
    
    # 保存结果
    with open(target_file, "w") as f:
        for item in processed_data:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    main()
