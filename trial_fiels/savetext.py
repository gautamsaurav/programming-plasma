import numpy as np
a=np.arange(1000)
froof=open("roof.txt")
ffloor=open("floor.txt")
with open("input.txt", "w") as f1:
    for line in froof:
        f1.write(line)
    froof.close()
    np.savetxt(f1, a)
    for line in ffloor:
        f1.write(line)
    ffloor.close()
