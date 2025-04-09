import random

cluster_size= 150
variance = 5
n_variance = 5

sequence = open("data1.txt", "r")

clusters = open("dataset1.txt", "w")

seq = sequence.readlines()
seq_count = len(seq)

for i in  range(0, seq_count):
    clusters.write(seq[i])
    clusters.write("*****************************\n")

    n = len(seq[i])
    print(n)


    for j in range(0, cluster_size):
        trace = list(seq[i])


        for k in range(0, variance):
            rn = random.randint(1,4)

            if rn == 1:
                change = 'A'
            elif rn == 2:
                change = 'C'
            elif rn == 3:
                change = 'G'
            elif rn == 4:
                change = 'T'

            rn2 = random.randint(0, n-2)
            trace[rn2] = change
    
        trace_str = "".join(trace)
        clusters.write(trace_str)

    clusters.write("\n\n")



sequence.close()
clusters.close()