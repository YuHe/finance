import os
from dashscope import Generation
import dashscope 

# 若使用新加坡地域的模型，请释放下列注释
# dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
messages = [{"role": "user", "content": "今天天气怎么样"}]


try:
    completion = Generation.call(
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key = "sk-xxx",
        api_key="sk-db6f93853954421ba885917e5ff9755a",
        # 可按需更换为其它深度思考模型
        model="qwen3.5-plus",
        messages=messages,
        result_format="message",
        enable_thinking=False,
        stream=True,
        incremental_output=False
    )
    if completion is None:
        raise ValueError("API调用返回空响应，请检查API Key和服务状态")

    # 由于关闭了思考功能，直接获取回复内容
    answer_content = ""
    for chunk in completion:
        if hasattr(chunk, 'output') and hasattr(chunk.output, 'choices') and chunk.output.choices:
            if hasattr(chunk.output.choices[0].message, 'content') and chunk.output.choices[0].message.content:
                print(chunk.output.choices[0].message.content, end="", flush=True)
                answer_content += chunk.output.choices[0].message.content

    print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
    print(answer_content)

except Exception as e:
    print(f"API调用失败: {e}")
    exit(1)