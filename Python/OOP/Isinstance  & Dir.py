# isinstance & dir
class Employee: 
    def __init__(self, name, salary, department): 
        self.name = name
        self.salary = salary
        self.department = department
        
    def showData(self):
        print("Name = {}".format(self.name))
        print("Salary = {}".format(self.salary))
        print("Department = {}".format(self.department))
        

obj1 = Employee("Thumma", 50000, "IT")
obj1.salary = 70000
obj1.showData()

print(isinstance(obj1, Employee))
print(dir(obj1))
