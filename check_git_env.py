import subprocess
import os

def check_env_in_git():
    print("🔍 正在检查 Git 追踪状态...")
    
    # 确保当前目录是 Git 仓库根目录
    if not os.path.exists(".git"):
        print("❌ 当前目录不是 Git 仓库，请确保你在项目根目录下运行。")
        return

    try:
        # 获取所有被 Git 追踪的文件列表
        result = subprocess.run(['git', 'ls-files'], capture_output=True, text=True, check=True)
        tracked_files = result.stdout.splitlines()
        
        # 检查 .env 是否在列表中
        if '.env' in tracked_files:
            print("🚨🚨🚨 【危险】发现 .env 已被 Git 追踪！")
            print("👉 请立刻在终端执行以下两条命令修复：")
            print("   1. git rm --cached .env")
            print("   2. git commit -m 'fix: 移除 .env 文件'")
        else:
            print("✅✅✅ 【非常安全】未发现 .env 被 Git 追踪！")
            print("👉 你的 API Key 非常安全，可以放心 Push。")
            
    except Exception as e:
        print(f"执行出错：{e}")

if __name__ == "__main__":
    check_env_in_git()