# tuple
product = ("Pant", 150.0, 10)
print(type(product))
print(len(product))
name, price, stock = product
for item in product:
    print(item)
print(product[0])
print(product)

colors1 = ("Red", "Green", "Blue")
print(colors1[:3])
print(colors1.index("Green"))
colors2 = tuple(("Black", "White"))
fullcolors = colors1 + colors2
print(fullcolors * 2)
