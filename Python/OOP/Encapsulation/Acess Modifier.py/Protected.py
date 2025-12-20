# protected
class Employee: 
    def __init__(self, name, salary, department): 
        # protected attribute
        self._name = name
        self._salary = salary
        self._department = department
        self._showData()
        
    # protected method
    def _showData(self):
        print("Name = {}".format(self._name))
        print("Salary = {}".format(self._salary))
        print("Department = {}".format(self._department))
        

obj1 = Employee("Thumma", 50000, "IT")
print(obj1._name)