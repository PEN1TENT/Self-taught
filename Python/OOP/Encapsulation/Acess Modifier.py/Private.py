# private
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
        

obj1 = Employee("Thumma", 50000, "IT")
# can't modify / using method