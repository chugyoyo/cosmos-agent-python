# test_cases/conftest.py
import pytest
from common.rest_client import RestClient

# 假设这是被测系统的地址 (可以使用 httpbin 模拟)
BASE_URL = "https://httpbin.org"

@pytest.fixture(scope="session")
def api_client():
    """
    全局 Fixture：在整个测试会话开始时执行一次
    返回一个初始化好的 RestClient 实例
    """
    client = RestClient(BASE_URL)
    # 如果需要登录，可以在这里先调登录接口
    # client.post("/login", json={"user": "admin", "pwd": "123"})
    yield client
    # yield 后面可以写清理操作，比如关闭数据库连接
    print("\n--- 所有测试执行完毕 ---")