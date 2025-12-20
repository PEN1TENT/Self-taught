# determine attribute
class Employee: 
    def detail(self, name, salary):
        self.name = name # attribute
        self.salary = salary
        
    def showData(self):
        print("Name = {}".format(self.name))
        print("Salary = {}".format(self.salary))
        

obj1 = Employee()
obj1.detail("Thumma", 50000)
obj1.showData()

obj2 = Employee()
obj2.detail("Jojo", 40000)
obj2.showData()