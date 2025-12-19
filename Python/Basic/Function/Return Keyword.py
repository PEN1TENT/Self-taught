# return keyword
balance = 1000 # global
def displayBalance():
    print("ยอดเงินคงเหลือในบัญชี", balance, "บาท")

def deposit(value):
    global balance
    money = value # local
    if (money <= 0):
        print("ไม่สามารถฝากเงินได้")
        return
    print("ฝากเงินจำนวน", money, "บาท")
    balance += money
    
def withdraw(value):
    global balance
    amount = value
    if (amount <= 0 or amount > balance):
        print("ไม่สามารถถอนเงินได้")
        return
    print("ถอนเงินจำนวน:", amount, "บาท")
    balance -= amount
    
displayBalance()
deposit(500)
displayBalance()