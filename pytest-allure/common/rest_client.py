# common/rest_client.py
import requests
import logging


class RestClient:
    def __init__(self, api_root_url):
        self.api_root_url = api_root_url
        self.session = requests.Session()  # 使用 Session 自动管理 Cookie

    def request(self, method, url, **kwargs):
        full_url = self.api_root_url + url
        try:
            # 记录请求日志（实际项目中这里会更详细）
            logging.info(f"正在请求: {method} {full_url}")
            if "json" in kwargs:
                logging.info(f"请求参数: {kwargs['json']}")

            response = self.session.request(method, full_url, **kwargs)

            logging.info(f"响应状态码: {response.status_code}")
            return response
        except Exception as e:
            logging.error(f"请求异常: {e}")
            raise

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)