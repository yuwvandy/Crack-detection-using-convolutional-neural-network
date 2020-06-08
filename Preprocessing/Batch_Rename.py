import os

Rawpath = 'D:/MATLAB_Undergraduate Design/数据/新数据/Test/'
Newpath = 'D:/MATLAB_Undergraduate Design/数据/新数据/Val/'
f = os.listdir(Rawpath)

p = 0
n = 201 
for i in f:
    oldname = Rawpath+f[p]
    newname = Newpath + "UnCrack" + str(n) + ".jpg"
    os.rename(oldname,newname)
    print(oldname,“======>",newname)
    p+=1
    n+=1