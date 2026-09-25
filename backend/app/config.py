"""\n配置管理\n统一从项目根目录的 .env 文件加载配置\n"""

import os
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
# 路径: MiroFish/.env (相对于 backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # 如果根目录没有 .env，尝试加载环境变量（用于生产环境）
    load_dotenv(override=True)


class Config:
    """Flask配置类"""
    
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mirofish-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # JSON配置 - 禁用ASCII转义，让中文直接显示
    JSON_AS_ASCII = False
    
    # LLM配置（统一使用OpenAI格式）
    LLM_API_KEY = os.environ.get('LLM_API_KEY')
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'gpt-4o-mini')
    
    # Zep配置
    ZEP_API_KEY = os.environ.get('ZEP_API_KEY')
    GRAPH_BACKEND = os.environ.get("GRAPH_BACKEND", "zep").strip().lower()

    # Local graph store (GRAPH_BACKEND=local): one SQLite file per graph.
    GRAPH_DATA_DIR = os.environ.get(
        "GRAPH_DATA_DIR", os.path.join(os.path.dirname(__file__), "../uploads/graphs")
    )
    GRAPH_EMBEDDER = os.environ.get("GRAPH_EMBEDDER", "http").strip().lower()
    GRAPH_EXTRACTOR = os.environ.get("GRAPH_EXTRACTOR", "stub").strip().lower()
    EMBED_BASE_URL = os.environ.get("EMBED_BASE_URL", "http://localhost:8001/v1")
    EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "intfloat/multilingual-e5-small")
    EMBED_API_KEY = os.environ.get("EMBED_API_KEY")
    # None = pick from the model name (e5 uses "query: " / "passage: ").
    EMBED_QUERY_PREFIX = os.environ.get("EMBED_QUERY_PREFIX")
    EMBED_PASSAGE_PREFIX = os.environ.get("EMBED_PASSAGE_PREFIX")

    # System One decisions (Jev-compatible). local = logit readout on the
    # local model server; http = POST {base}/v1/systemone.
    SYSTEM_ONE_BACKEND = os.environ.get("SYSTEM_ONE_BACKEND", "local").strip().lower()
    SYSTEM_ONE_BASE_URL = os.environ.get("SYSTEM_ONE_BASE_URL", "http://localhost:8000/v1")
    SYSTEM_ONE_MODEL = os.environ.get("SYSTEM_ONE_MODEL") or os.environ.get("LLM_MODEL_NAME", "qwen3.5-4b")
    SYSTEM_ONE_API_KEY = os.environ.get("SYSTEM_ONE_API_KEY")
    SYSTEM_ONE_TOP_K = int(os.environ.get("SYSTEM_ONE_TOP_K", "20"))
    SYSTEM_ONE_PROMPT_FORMAT = os.environ.get("SYSTEM_ONE_PROMPT_FORMAT", "chatml").strip().lower()
    
    # 文件上传配置
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown'}
    
    # 文本处理配置
    DEFAULT_CHUNK_SIZE = 500  # 默认切块大小
    DEFAULT_CHUNK_OVERLAP = 50  # 默认重叠大小
    
    # OASIS模拟配置
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')
    
    # OASIS平台可用动作配置
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'DO_NOTHING', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'DO_NOTHING', 'FOLLOW', 'MUTE'
    ]
    
    # Report Agent配置
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))
    
    @classmethod
    def zep_key_missing(cls) -> bool:
        """True only when the zep graph backend is selected without a key."""
        return cls.GRAPH_BACKEND == "zep" and not cls.ZEP_API_KEY

    @classmethod
    def validate(cls) -> list[str]:
        """验证必要配置"""
        errors: list[str] = []
        if not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY 未配置")
        if cls.GRAPH_BACKEND not in ("zep", "local"):
            errors.append("GRAPH_BACKEND 必须是 zep 或 local")
        elif cls.GRAPH_BACKEND == "zep":
            if not cls.ZEP_API_KEY:
                errors.append("ZEP_API_KEY 未配置")
            if os.environ.get("ZEP_API_URL"):
                errors.append("ZEP_API_URL 不受支持；MiroFish 仅连接 Zep Cloud")
        if cls.DEBUG:
            import warnings
            warnings.warn("Flask DEBUG mode is enabled. Do not use in production.", RuntimeWarning)
        return errors
