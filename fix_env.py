import subprocess
import sys

def install_and_check():
    print("正在为 Python 环境安装必要依赖...")
    try:
        # 安装 openpyxl 以读取 .xlsx
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl", "pandas"])
        print("\n\u2713 依赖安装成功！")
        
        # 验证环境
        import pandas as pd
        import openpyxl
        print(f"\u2713 环境验证通过: Pandas {pd.__version__}, openpyxl 引擎已就绪。")
        print("\n现在您可以重新运行 visualize_cases.py 了。")
    except Exception as e:
        print(f"\n\u2717 安装失败: {e}")
        print("请尝试在终端手动运行: pip install openpyxl pandas")

if __name__ == "__main__":
    install_and_check()
