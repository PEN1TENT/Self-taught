# object to string
class Employee: 
    minSalary = 12000
    maxSalary = 100000
    companyName = "XYZ"
    def __init__(self, name, salary, department): 
        self.__name = name
        self.__salary = salary
        self.__department = department
        self._showData()
        
    def _showData(self):
        print("Name = {}".format(self.__name))
        print("Salary = {}".format(self.__salary))
        print("Department = {}".format(self.__department))
        
    def _getIncome(self):
        return self.__salary * 12
        
    def __str__(self):
        return("Name = {}, Department = {}, Salary = {}, Annual = {}".format(self.__name, self.__department, self.__salary, self._getIncome()))
class Accounting(Employee):
    __departmentName = "Accounting"
    def __init__(self, name, salary):
        super().__init__(name, salary, self.__departmentName)
        super()._showData()

class Programmer(Employee):
    __departmentName = "Programmer"
    def __init__(self, name, salary, department):
        super().__init__(name, salary, self.__departmentName)


    
accounting = Accounting("Thumma", 50000)
print(accounting.__str__())

