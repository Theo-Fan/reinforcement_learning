"""
使用 Process 子类创建进程
"""

from multiprocessing import Process
import os, time

class SubProcess(Process):
    def __init__(self, name):
        super().__init__()
        self.name = name
        
    
    def run(self):
        print(f"sub_process name: {self.name}, pid: {os.getpid()}, father pid: {os.getppid()}")


if __name__ == '__main__':
    print("father process starting")
    
    lst = []
    
    for i in range(5):
        p = SubProcess(f"process: {i}")
        p.start()
        lst.append(p)
        
    for i in lst:
        i.join()
    
    print("father process finished")
        