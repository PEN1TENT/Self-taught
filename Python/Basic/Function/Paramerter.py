# parameter
def sayHello(time, username, age): # parameters
    print("่สวัสดี", time, username)
    print("ปีนี้คุณมีอายุ", age ,"ปี")

def showTable(num):
    print("----------")
    for i in range (1,13):
        print(f"{num} x {i} = {num*i}")
    
sayHello("ตอนเช้า", "ธรรมะ", 19) # arguments
showTable(3)