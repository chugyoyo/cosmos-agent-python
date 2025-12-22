# test_cases/test_order.py
import pytest
import allure


@allure.feature("订单模块")  # 一级分类
class TestOrder:

    @allure.story("创建订单-正向用例")  # 二级分类
    @allure.title("测试用户成功下单")  # 用例标题
    @allure.severity(allure.severity_level.CRITICAL)  # 优先级
    def test_create_order_success(self, api_client):
        """
        这里写用例的详细描述：
        1. 构造下单数据
        2. 发起 POST 请求
        3. 断言响应码和关键字段
        """
        # 1. 准备数据
        payload = {
            "sku_id": "10086",
            "quantity": 1,
            "price": 99.9
        }

        # 2. 调用步骤 (显示在报告中)
        with allure.step("步骤1: 调用创建订单接口"):
            # 这里用 post 模拟下单，httpbin/post 会返回我们传的数据
            res = api_client.post("/post", json=payload)

        # 3. 断言
        with allure.step("步骤2: 校验响应结果"):
            assert res.status_code == 200
            assert res.json()['json']['sku_id'] == "10086"

    @allure.story("查询订单")
    @pytest.mark.parametrize("order_id", [1001, 1002])  # 参数化运行多条数据
    def test_get_order(self, api_client, order_id):
        with allure.step(f"查询订单ID: {order_id}"):
            res = api_client.get("/get", params={"id": order_id})
            assert res.status_code == 200