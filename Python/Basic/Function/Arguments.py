# args / kwargs
# def saveEmployee(*data): #args
#     print(f"ชื่อพนักงาน {data[0]}, แผนก {data[1]}")
#     print(f"เงินเดืิอนื {data[2]} บาท")
#     print(f"ที่อยู่ {data[3]} ")
#     print("----------------")


def saveEmployee(**data):  # kwargs
    print(f"ชื่อพนักงาน {data['name']}, แผนก {data['department']}")
    print(f"เงินเดืิอนื {data['salary']} บาท")
    print(f"ที่อยู่ {data['province']} ")
    print("----------------")


saveEmployee(name="ก้อง", department="ไอที", salary=30000, province="ภูเก็ต")
