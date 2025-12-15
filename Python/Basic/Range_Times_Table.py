# range times table
start = int(input("Start : "))
end = int(input("End : "))

for number in range(start, end + 1):
    print("Times Table", number)
    print("-------------------")
    for i in range(1, 13):
        print(number, "x", i, "=", number * i)
    print("-------------------")
