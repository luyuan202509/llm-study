import multiprocessing as mp
import os

def worker(num):
    """子进程要执行的任务"""
    print(f"子进程 {num}: PID={os.getpid()}, 父进程PID={os.getppid()}")
    return num * num

if __name__ == "__main__":
    # 设置启动方法为 spawn
    mp.set_start_method('spawn')
    
    print(f"主进程 PID={os.getpid()}")
    
    # 创建进程池
    with mp.Pool(processes=2) as pool:
        results = pool.map(worker, [1, 2, 3, 4])
        print(f"结果: {results}")