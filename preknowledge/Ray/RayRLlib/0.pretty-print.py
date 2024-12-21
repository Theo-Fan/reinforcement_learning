import pprint

"""
pretty-print: 主要用于以一种更易读的形式输出数据
"""
data = {
    'name': 'Alice',
    'age': 30, 
    'children': [
        {'name': 'Bob', 'age': 10},
        {'name': 'Charlie', 'age': 8}
    ],
    'pets': {'dog': 'Rex', 'cat': 'Whiskers'},
    'hobbies': ['reading', 'hiking', 'gardening']
}

pprint.pprint(data)
