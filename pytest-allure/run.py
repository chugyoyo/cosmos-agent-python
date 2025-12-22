# run.py
import pytest
import os

if __name__ == '__main__':
    # 1. 执行 pytest 测试
    pytest.main()

    # 2. 如果安装了 allure 命令行工具，自动生成 HTML 报告
    # 注意：需要在系统环境变量中配置好 allure 的 bin 目录
    # os.system("allure generate ./report/xml -o ./report/html --clean")

    print("测试执行完成")