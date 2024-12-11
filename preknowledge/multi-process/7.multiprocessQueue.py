from multiprocessing import Queue

"""
put(obj, block=True, timeout): 添加元素，当队满时阻塞，超过 timeout 时间后抛出异常
put_nowait(obj) 相当于 put(obj, False) 队满直接报错

"""

if __name__ == '__main__':
    q = Queue(3)
    
    print(f"queue is none? {q.empty()}")
    print(f"queue is full? {q.full()}")
    # print(f"queue size is {q.qsize()}") # mocOS 上未实现 NotImplementedError
    
    q.put("hello")
    q.put("world")
    
    
    print()
    print(f"queue is none? {q.empty()}")
    print(f"queue is full? {q.full()}")
    # print(f"queue size is {q.qsize()}")
    
    
    q.put("python")
    
    
    print()
    print(f"queue is none? {q.empty()}")
    print(f"queue is full? {q.full()}")
    # print(f"queue size is {q.qsize()}"
    
    
    q.put("other", block=True, timeout=2) # 此时队列已满，put会阻塞, 等待超过 2s 则报错
    