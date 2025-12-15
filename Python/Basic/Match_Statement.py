# match statement
service = input("Enter Service Number (1-3): ")

match service:
    case "1":
        print("Withdraw")
    case "2":
        print("Deposit")
    case "3":
        print("Check Balance")
    case "":
        print("Invalid Service Number")
