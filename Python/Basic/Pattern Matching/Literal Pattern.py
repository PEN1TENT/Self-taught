# literal pattern
service = None
match service:
    case 1:
        print("ฝากเงิน")
    case 2:
        print("ถอนเงิน")
    case 3:
        print("สอบถามยอดเงินคงเหลือ")
    case None:
        print("หมายเลขบริการไม่ถูกต้อง")
