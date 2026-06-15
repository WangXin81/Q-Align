from typing import Optional
import os
def get_api_key(
    env_name: str = "API_KEY",
    dotenv_path: str = ".env",
    required: bool = True,
) -> Optional[str]:
    """
    优先从环境变量读取 API key。
    如果环境变量不存在，再尝试从 .env 文件读取。
    """

    api_key = os.getenv(env_name)
    if api_key:
        return api_key.strip()

    env_file = Path(dotenv_path)
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)

            if key.strip() == env_name:
                api_key = value.strip().strip('"').strip("'")
                if api_key:
                    os.environ[env_name] = api_key
                    return api_key

    if required:
        raise RuntimeError(
            f"未找到 API key。请设置环境变量 {env_name}，或在 {dotenv_path} 中写入 {env_name}=你的key"
        )

    return None