#set
animals = {"Dog", "Cat", "Lion", "Tiger", True, 100}
print(type(animals))
print(len(animals))
for item in animals:
    print(item)
print("Dog" in animals)
animals.add("Duck")
animals.update(("Pig", "Elephant"))
print(animals)

pets = set(("Dog", "Cat", "Rabbit", "Hedgehog"))

data = animals.union(pets)
data = animals.intersection(pets)
data = animals.difference(pets)