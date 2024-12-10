from multiprocessing import Process
import os, time

def test():
    print(f"subprocessing id : {os.getpid()}, father processing id: {os.getppid()}")
    time.sleep(1)

if __name__ == '__main__':
    print("start father processing")
    lst = []
    
    for i in range(5):
        p = Process(target=test)
        p.start() # 进程开始执行
        lst.append(p)
        
    for i in lst: # 该语句用于阻塞父进程，不添加则print会先执行
        i.join()
    
    print("father processing finished")