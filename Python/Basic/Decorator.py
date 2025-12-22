# decorator
def div(a, b):
    print(a/b)

def smart_div(func):
    def inner(a, b):
        if a < b:
            a, b = b, a
        return func(a, b)
    
    return inner

divl = smart_div(div)
divl(2, 4)