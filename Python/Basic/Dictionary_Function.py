# dictionary function
colors = {"red": "แดง", "green": "เขียว", "blue": "น้ำเงิน"}
print(colors.keys())
print(colors.values())
print(colors.items())
for key in colors.keys():
    print(key)
for value in colors.value():
    print(value)
for key, value in colors.items():
    print(key, "=", value)
print(colors.get("red"))
colors.pop("blue")
colors["yellow"] = "สีเหลือง"
colors.update({"red": "แดงเข้ม"})
maincolor = colors.copy()
print(colors)
colors.clear()
