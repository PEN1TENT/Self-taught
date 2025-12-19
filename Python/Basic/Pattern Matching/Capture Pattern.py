# capture pattern
service = 10
match service:
    case 1:
        print("ฝากเงิน")
    case 2:
        print("ถอนเงิน")
    case 3:
        print("สอบถามยอดเงินคงเหลือ")
    case service:
        print(f"ไม่มีบริการหมายเลข {service} ในระบบ กรุณาทำรายการให้อีกครั้ง!")