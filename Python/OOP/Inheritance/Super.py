# super
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
    __departmentName = "Accounting"
    def __init__(self, name, salary):
        super().__init__(name, salary, self.__departmentName)
        super().__showData()

class Programmer(Employee):
    __departmentName = "Programmer"
    def __init__(self, name, salary, department):
        super().__init__(name, salary, self.__departmentName)


    
accounting = Accounting("Thumma", 50000)
print(accounting.companyName)

