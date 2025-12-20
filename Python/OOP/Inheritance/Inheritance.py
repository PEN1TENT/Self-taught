# inheritance
class Employee: 
    minSalary = 12000
    maxSalary = 100000
    companyName = "XYZ"
    def __init__(self, name, salary, department): 
        self.__name = name
        self.__salary = salary
        self.__department = department
        self.__showData()
        
    def __showData(self):
        print("Name = {}".format(self.__name))
        print("Salary = {}".format(self.__salary))
        print("Department = {}".format(self.__department))
        
class Accounting(Employee):
    def __init__(self):
        pass

class Programmer(Employee):
    pass

class Sale(Employee):
    pass
    
accounting = Accounting()
print(accounting.companyName)

