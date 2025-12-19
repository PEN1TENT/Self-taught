# wildcard pattern
service = 100
match service:
    case 1:
        print("ฝากเงิน")
    case 2:
        print("ถอนเงิน")
    case 3:
        print("สอบถามยอดเงินคงเหลือ")
    case _: # default case
        print("ข้อมูลไม่ถูกต้อง")