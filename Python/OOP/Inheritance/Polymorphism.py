# overloading & overriding method
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
    def __init__(self, name, salary, age):
        # overloading
        super().__init__(name, salary, self.__departmentName)
        self.__age = age
    
    # overriding
    def _showData(self):
        super()._showData()
        print("Age = {}".format(self.__age))
    
    def __str__(self):
        return(super().__str__()+", Age = {}".format(self.__age))

class Programmer(Employee):
    __departmentName = "Programmer"
    def __init__(self, name, salary, experience, skill):
        super().__init__(name, salary, self.__departmentName)
        self.__exp = experience
        self.__skill = skill
        
    def _showData(self):
        super()._showData()
        print("Experience = {}".format(self.__exp))
        print("Skill = {}".format(self.__skill))
    
    def __str__(self):
        return(super().__str__()+", Experience = {}, Skill = {}".format(self.__exp, self.__skill))
    
accounting = Accounting("Thumma", 50000, 19)
print(accounting.__str__())
programmer = Programmer("Jojo", 60000, 5, "Python")

