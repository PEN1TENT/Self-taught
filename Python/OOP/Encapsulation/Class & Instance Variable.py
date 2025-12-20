# class variable
class Employee: 
    # class variable
    _minSalary = 12000
    _maxSalary = 100000
    def __init__(self, name, salary, department): 
        # innstance variable
        self.__name = name
        self.__salary = salary
        self.__department = department
        self.__showData()
        
    def __showData(self):
        print("Name = {}".format(self.__name))
        print("Salary = {}".format(self.__salary))
        print("Department = {}".format(self.__department))
        
print(Employee._maxSalary)
obj1 = Employee("Thumma", 50000, "IT")
