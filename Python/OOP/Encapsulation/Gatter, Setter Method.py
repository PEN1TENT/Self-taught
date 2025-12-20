# gatter, setter method
class Employee: 
    def __init__(self, name, salary, department): 
        # private attribute
        self.__name = name
        self.__salary = salary
        self.__department = department
        self.__showData()
        
    # private method
    def __showData(self):
        print("Name = {}".format(self.__name))
        print("Salary = {}".format(self.__salary))
        print("Department = {}".format(self.__department))
        
    # setter method
    def setName(self, newname):
        self.__name = newname
    
    def setSalary(self, newsalary):
        self.__salary = newsalary
        
    def setDepartment(self, newdepartment):
        self.__department = newdepartment
        
    # getter method
    def getName(self):
        return self.__name
    
    def getSalary(self):
        return self.__salary
        
    def getDepartment(self):
        return self.__department

obj1 = Employee("Thumma", 50000, "IT")
obj1.setSalary("60000")
print("Name = ", format(obj1.getName()))