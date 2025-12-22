from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException
import time


# --- 核心函数：自动化 Agent ---
def auto_click_agent(url, target_selector, selector_type=By.CSS_SELECTOR):
    """
    使用 Selenium WebDriver 自动化访问指定 URL 并点击目标元素。

    :param url: 目标网页 URL。
    :param target_selector: 目标元素的定位符（例如 CSS 选择器或 XPath）。
    :param selector_type: 定位符类型，默认使用 By.CSS_SELECTOR。
    """

    print(f"Agent 启动中，目标 URL: {url}")

    # 1. 初始化 WebDriver 服务
    # 使用 Service 确保驱动程序可以被 Selenium Manager 自动处理
    # 在 Selenium 4.6+ 版本中，通常不需要手动指定驱动路径
    service = Service()

    # 2. 配置 WebDriver 选项 (可选：设置为无头模式)
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # 运行在后台，不显示浏览器窗口
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    # 3. 启动 Chrome 浏览器
    driver = webdriver.Chrome(service=service, options=options)

    try:
        # 4. 访问目标 URL
        driver.get(url)
        print(f"成功访问页面: {driver.title}")

        for _ in range(1000):

            # 增加等待时间，确保页面元素加载完毕
            time.sleep(1)

            # 5. 定位目标元素并执行点击
            print(f"尝试定位元素: {target_selector}")

            # driver.find_element(定位类型, 定位值)
            click_element = driver.find_element(selector_type, target_selector)

            # 执行点击操作
            click_element.click()
            print("✅ 元素点击成功！")

            # 输入文本
            driver.find_element(By.ID, "inputSupplierName").send_keys("test supplier name")

            # 提交
            driver.find_element(By.ID, "handleCreateSubmit").click()

            # 6. 等待并验证点击结果 (可选)
            # time.sleep(2)

            print(f"点击后的页面标题: {driver.title}")
    except NoSuchElementException:
        print(f"❌ 错误：未找到目标元素，定位符: {target_selector}")
    except Exception as e:
        print(f"发生其他错误: {e}")
    finally:
        # 7. 关闭浏览器
        driver.quit()
        print("Agent 任务完成，浏览器已关闭。")


# --- 示例运行 ---
if __name__ == '__main__':
    # 假设我们要点击百度搜索页面的“百度一下”按钮
    target_url = "http://localhost/InventoryInOutOrder"

    # 如何确定定位符？
    # 在 Chrome 中右键目标元素 -> 检查 -> 选中元素后，右键 -> Copy -> Copy selector 或 Copy XPath

    # 示例定位符：百度搜索按钮的 id 是 'su'
    # 使用 By.ID 定位
    auto_click_agent(
        url=target_url,
        target_selector="createInOrder",
        selector_type=By.ID
    )

    # 示例 2：使用 CSS 选择器定位（更灵活）
    # auto_click_agent(
    #     url=target_url,
    #     target_selector="#su",  # '#' 表示 ID 选择器
    #     selector_type=By.CSS_SELECTOR
    # )