#Mathmatic Operator
score = int(input("Enter your score:"))
print("Your score is ", score)
grade = None

if score >= 80 and score <= 100:
    grade = "A"
elif score >= 70 and score <= 79:
    grade = "B"
elif score >= 0 and score <= 69:
    grade = "F"
else:
    grade = "N (Invalid)"

print("Your grade is ", grade)
