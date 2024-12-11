from multiprocessing import Process
import os, time

def sub_process1(name):
    print(f"subpid: {os.getpid()}, father_pid: {os.getppid()}, -------{name}")
    time.sleep(1)
    

def sub_process2(name):
    print(f"subpid: {os.getpid()}, father_pid: {os.getppid()}, -------{name}")
    time.sleep(1)

"""

Process(group=None, target, name, args, kwargs)

params:
    group:  表示分组，实际上不用
    target: 表示子进程要执行的任务，支持函数名
    name:   表示自己撑的名称
    arg:    表示调用函数的位置参数，以元组的形式进行传递
    kwargs: 表示调用函数的关键词参数，以字典方式进行传递

"""

if __name__ == '__main__':
    print("father process starting")
    for i in range(5):
        p1 = Process(target=sub_process1, args=('theo',))
        
        p2 = Process(target=sub_process2, args=("aaaa",)) # 子进程执行 target 传入的方法
        
        # p1 = Process()
        # p2 = Process() # 没有指定 target 方法则调用父进程的 run 函数（继承）
        
        p1.start()
        p2.start()
        
        # print(f"{p1.name} is running {p1.is_alive()}") # is_alive() 判断该进程是否还在运行
        # print(f"{p2.name} is running {p2.is_alive()}")
        
        print(p1.name, " pid: ", p1.pid)
        print(p2.name, " pid: ", p2.pid)
        

        
        # print(f"{p1.name} is running {p1.is_alive()}") 
        # print(f"{p2.name} is running {p2.is_alive()}")
        
        
        p1.terminate() # 强制终止进程
        
        
        # p1.join() # 主进程要等 p1 执行结束，阻塞主进程
        p2.join()
        
        
        print()
        print()
        
    
    print("father process finished")
        
        
        
        
