from incl import printer
from pathlib import Path

for i in range(5):
    print(i)
print("Hello World Python!")
someVar = 4
if someVar > 3:
    print("heeere we go")

def someFunc(arg):
    myList = []
    print(type(arg))
    print(type(arg) == str)
    if type(arg) == str:
        myList = arg.split()
        print(myList)

someFunc('Hello World Yo')

def fact(n):
    if n == 1:
        return n
    return n*fact(n - 1)

print(fact(5))

# write some bubble sort functio in py
def bubble():
    return

printer()

p = Path('.')
print(list(p.glob('*.py')))

print(list(range(5,5)))