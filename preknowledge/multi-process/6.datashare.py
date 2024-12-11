# 多个进程之间的数据是否共享【否】

from multiprocessing import Process

a = 100

def add():
    print("subprocess start")
    global a
    a += 30
    print(f"a = {a}")
    print("subprocess finished")
    

def sub():
    print("subprocess start")
    global a
    a -= 50
    print(f"a = {a}")
    print("subprocess finished")
    

if __name__ == '__main__':
    print("father process start")
    print(f"init a = {a}")
    p1 = Process(target=add)
    p2 = Process(target=sub)
    
    p1.start()
    p2.start()
    
    p1.join()
    p2.join()
    
    print("father process finished")
    print(f"end a = {a}")