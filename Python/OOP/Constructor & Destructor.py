# constructor & destructor
class Employee: 
    def __init__(self, name, salary, department): # Constructor
        self.name = name
        self.salary = salary
        self.department = department
        
    def showData(self):
        print("Name = {}".format(self.name))
        print("Salary = {}".format(self.salary))
        print("Department = {}".format(self.department))
    
    def __del__(self): # Desturctor
        print("Call Destructor")
        

obj1 = Employee("Thumma", 50000, "IT")
obj1.salary = 70000
obj1.showData()
