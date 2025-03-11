import subprocess
import time
import argparse

def check_gpus_idle():
    try:
        # 使用 nvidia-smi 命令获取 GPU 状态
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                                stdout=subprocess.PIPE, text=True)
        output = result.stdout.strip().splitlines()

        # 设置显存使用阈值（例如 10 MB，低于该值视为空闲）
        memory_threshold = 100

        # 检查所有 GPU 的显存使用情况
        for memory_used in output:
            if int(memory_used) > memory_threshold:  # 如果显存使用大于阈值，视为不空闲
                return False

        return True  # 所有 GPU 空闲
    except Exception as e:
        print(f"Error checking GPU status: {e}")
        return False

def main():
    print("开始监控 GPU 使用情况...")
    while True:
        if check_gpus_idle():
            print("两张 GPU 都空闲！")
            try:
                # 执行指定的命令
                subprocess.run(['nohup', 'bash', 'run.sh'])
                print("命令已启动。")
            except Exception as e:
                print(f"Error executing command: {e}")
            break  # 如果需要可以选择继续监控或退出

        time.sleep(10)  # 每 10 秒检查一次

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Process some arguments.")
    
    # 添加命令行参数
    # parser.add_argument("kind", type=str, help="训练的是啥？")

    
    # 解析参数
    # args = parser.parse_args()
    
    # 将参数传递给 main 函数
    # main(args.kind)

    main()
