# attribute and method
class Employee: 
    # Create method
    def detail(self):
        self.name = "Thumma"
        self.salary = 50000
        print("Name = {}".format(self.name))
        print("Salary = {}".format(self.salary))

emp1 = Employee()
emp1.detail()