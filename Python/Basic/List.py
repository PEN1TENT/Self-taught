# List
product = ["Pant", 99.99, 10, True]
print(type(product))
print(len(product))
print(product[0])
for element in product:
    print(element)
print(product)
product[0] = "Shirt"

colors1 = ["Black", "Red", "Green", "Black"]
colors2 = list(("White", "Blue", "Orange"))
fullcolors = colors1 + colors2
print(fullcolors)
