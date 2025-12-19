# raise
try:
    number1 = int(input("ป้อนตัวเลขที่ 1:"))
    number2 = int(input("ป้อนตัวเลขที่ 2:"))
    if number1 < 0 or number2 < 0:
        raise Exception("ข้อมูลตัวเลขต้อเป็นค่าบวกเท่านั้น!")
        
    result = number1/number2
    print("ผลลัทธ์ =", result)
except Exception:
    print("ข้อมูลไม่ถูกต้อง")