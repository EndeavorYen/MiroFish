"""
LLM客户端封装
统一使用OpenAI格式调用
"""

import json
import re
import time
from typing import Optional, Dict, Any, List
from openai import OpenAI

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger('mirofish.llm')

# 预编译正则：避免每次调用重新编译
_RE_THINK_TAG = re.compile(r'<think>[\s\S]*?</think>')
_RE_MD_FENCE_HEAD = re.compile(r'^```(?:json)?\s*\n?', re.IGNORECASE)
_RE_MD_FENCE_TAIL = re.compile(r'\n?```\s*$')


def fix_truncated_json(content: str) -> str:
    """修复被截断的JSON（闭合未匹配的括号）"""
    content = content.strip()

    open_braces = content.count('{') - content.count('}')
    open_brackets = content.count('[') - content.count(']')

    # 如果末尾不是合法JSON终止字符，尝试闭合字符串
    if content and content[-1] not in '",}]':
        content += '"'

    content += ']' * max(open_brackets, 0)
    content += '}' * max(open_braces, 0)

    return content


class LLMClient:
    """LLM客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME
        self.api_key = Config.get_llm_api_key(api_key=api_key, base_url=self.base_url)

        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")

        self.timeout = Config.get_llm_timeout_seconds(self.base_url)
        self.is_local = Config.is_local_openai_compatible_base_url(self.base_url)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

    def _resolve_think(self, think: Optional[bool]) -> bool:
        """解析 think 参数：None 使用 Config 预设，否则使用指定值"""
        return Config.LLM_THINK if think is None else think

    def _build_kwargs(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        response_format: Optional[Dict],
        think: Optional[bool]
    ) -> Dict[str, Any]:
        """构建 completions.create 的参数"""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if self.is_local:
            # Ollama 0.17+ 本地端点特殊处理：
            # 1. 不传 response_format（json_object 会干扰 thinking 模型的输出）
            # 2. 不传 max_tokens（Ollama 将 thinking + content 合计算，thinking 会吃光额度）
            # 3. 不传 extra_body（{"think": false} 会被 Ollama 0.17 误读为开启 thinking）
            # 4. 通过 /no_think prompt 指令控制 thinking（Qwen 模型原生支持）
            if not self._resolve_think(think):
                messages = self._inject_no_think(messages)
                kwargs["messages"] = messages
        else:
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if response_format:
                kwargs["response_format"] = response_format
        return kwargs

    @staticmethod
    def _inject_no_think(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """注入 /no_think 指令禁用 thinking（Qwen 模型原生支持）

        在 system message 末尾 + 第一条 user message 开头注入，
        避免修改多轮对话中的历史消息。
        """
        messages = [m.copy() for m in messages]
        user_injected = False
        for m in messages:
            if m["role"] == "system":
                m["content"] = m["content"].rstrip() + "\n/no_think"
            elif m["role"] == "user" and not user_injected:
                m["content"] = "/no_think\n" + m["content"]
                user_injected = True
        return messages

    @staticmethod
    def _clean_content(content: str) -> str:
        """清理 LLM 回应中的 think 标签和 markdown 代码块"""
        content = _RE_THINK_TAG.sub('', content).strip()
        content = _RE_MD_FENCE_HEAD.sub('', content)
        content = _RE_MD_FENCE_TAIL.sub('', content)
        return content.strip()

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
        think: Optional[bool] = None
    ) -> str:
        """
        发送聊天请求

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（如JSON模式）
            think: 是否启用思考模式（None=使用Config预设, True=强制开启, False=强制关闭）

        Returns:
            模型响应文本
        """
        kwargs = self._build_kwargs(messages, temperature, max_tokens, response_format, think)
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        return self._clean_content(content)

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        think: Optional[bool] = False,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回JSON

        内置截断侦测、JSON修复、重试（温度递减）机制。
        不设置 max_tokens 上限，让 LLM 自由发挥。
        默认 think=False（结构化输出不需要思考，省token、快速回应）。

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数（默认None=不限制）
            think: 是否启用思考模式（默认False=关闭，适合结构化输出）
            max_retries: 最大重试次数

        Returns:
            解析后的JSON对象
        """
        last_error = None
        kwargs = self._build_kwargs(
            messages, temperature, max_tokens,
            {"type": "json_object"}, think
        )

        for attempt in range(max_retries + 1):
            try:
                # 每次重试降低温度，提高输出稳定性
                kwargs["temperature"] = max(temperature - (attempt * 0.1), 0.0)

                response = self.client.chat.completions.create(**kwargs)
                raw_content = response.choices[0].message.content or ""
                finish_reason = response.choices[0].finish_reason

                # Debug: 记录原始返回，帮助诊断空内容问题
                if not raw_content:
                    reasoning = getattr(response.choices[0].message, "reasoning", None) or ""
                    logger.warning(
                        f"LLM content为空 (attempt {attempt+1}): "
                        f"finish_reason={finish_reason}, "
                        f"reasoning_len={len(reasoning)}"
                    )

                content = raw_content

                # 截断侦测与修复
                if finish_reason == 'length':
                    logger.warning(f"LLM JSON输出被截断 (attempt {attempt+1})，尝试修复...")
                    content = fix_truncated_json(content)

                content = self._clean_content(content)

                if not content:
                    raise json.JSONDecodeError("LLM返回空内容", "", 0)

                return json.loads(content)

            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(f"JSON解析失败 (attempt {attempt+1}/{max_retries+1}): {str(e)[:100]}")
                if attempt < max_retries:
                    continue

            except Exception as e:
                last_error = e
                logger.warning(f"LLM调用失败 (attempt {attempt+1}/{max_retries+1}): {str(e)[:100]}")
                if attempt < max_retries:
                    time.sleep(2 * (attempt + 1))
                    continue

        raise ValueError(f"LLM返回的JSON格式无效（{max_retries+1}次尝试后）: {str(last_error)}")
