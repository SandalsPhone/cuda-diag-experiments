import random

count = 2000
n = 100

f = open("data1.txt", "w")

for t in range(0, count):
    for j in range(0, n):
        rn = random.randint(1,4)

        if rn == 1:
            f.write("A")
        elif rn == 2:
            f.write("C")
        elif rn == 3:
            f.write("G")
        elif rn == 4:
            f.write("T")
    
    f.write("\n")


f.close()
