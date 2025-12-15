# nested-if
username = input("Enter your username: ")
password = input("Enter your password: ")

if username == "member" and password == "1234":
    print("Login Successed")
    service = input("Enter Service number (1-2): ")
    if service == "1":
        print("Withdraw")
    elif service == "2":
        print("Deposit")
    else:
        print("Invaid Service Number")
else:
    print("Not Found User")
