# find infinite input sum
total = 0
while True:
    number = int(input("Enter Number: "))
    if number <= 0:
        break
    total += number
print("Sum =", total)
