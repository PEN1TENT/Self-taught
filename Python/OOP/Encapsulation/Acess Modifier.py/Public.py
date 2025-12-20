# public
class Employee: 
    def __init__(self, name, salary, department): 
        # public attribute
        self.name = name
        self.salary = salary
        self.department = department
        
    # public method
    def showData(self):
        print("Name = {}".format(self.name))
        print("Salary = {}".format(self.salary))
        print("Department = {}".format(self.department))
        

obj1 = Employee("Thumma", 50000, "IT")
