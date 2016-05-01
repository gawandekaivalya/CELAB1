# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 16:30:35 2016

@author: Kaivalya
"""
import matplotlib.pyplot

Z = [[3,0,1,0,1,2,3,0,2,1],
     [1,2,2,2,3,1,0,0,3,2],
     [2,0,0,3,1,2,1,0,3,3],
     [3,0,2,0,2,0,3,2,3,0],
     [0,3,3,1,1,3,2,3,2,0],
     [2,1,1,3,0,2,3,0,1,1],
     [2,2,2,0,3,1,0,1,0,0],
     [3,0,3,2,2,0,0,2,2,2],
     [0,3,2,3,0,1,0,3,1,0],
     [1,2,0,0,0,1,2,0,3,3]]
     
def neighbours0(Z):
    R,C = len(Z), len(Z[0])
    n0 = [[0,]*(C)  for i in range(R)]
    for x in range(1,C-1):
        for y in range(1,R-1):
            Neigh=[Z[x-1][y-1],Z[x][y-1],Z[x+1][y-1],Z[x-1][y],Z[x+1][y],Z[x-1][y+1],Z[x][y+1],Z[x+1][y+1]]
            i=0
            for n in Neigh:
                if n==0:
                    i=i+1
                n0[x][y]=i
    return n0
     
def neighbours1(Z):
    R,C = len(Z), len(Z[0])
    n1 = [[0,]*(C)  for i in range(R)]
    for x in range(1,C-1):
        for y in range(1,R-1):
            Neigh=[Z[x-1][y-1],Z[x][y-1],Z[x+1][y-1],Z[x-1][y],Z[x+1][y],Z[x-1][y+1],Z[x][y+1],Z[x+1][y+1]]
            i=0
            for n in Neigh:
                if n==1:
                    i=i+1
                n1[x][y]=i
    return n1
     
def neighbours2(Z):
    R,C = len(Z), len(Z[0])
    n2 = [[0,]*(C)  for i in range(R)]
    for x in range(1,C-1):
        for y in range(1,R-1):
            Neigh=[Z[x-1][y-1],Z[x][y-1],Z[x+1][y-1],Z[x-1][y],Z[x+1][y],Z[x-1][y+1],Z[x][y+1],Z[x+1][y+1]]
            i=0
            for n in Neigh:
                if n==2:
                    i=i+1
                n2[x][y]=i
    return n2

def neighbours3(Z):
    R,C = len(Z), len(Z[0])
    n3 = [[0,]*(C)  for i in range(R)]
    for x in range(1,C-1):
        for y in range(1,R-1):
            Neigh=[Z[x-1][y-1],Z[x][y-1],Z[x+1][y-1],Z[x-1][y],Z[x+1][y],Z[x-1][y+1],Z[x][y+1],Z[x+1][y+1]]
            i=0
            for n in Neigh:
                if n==3:
                    i=i+1
                n3[x][y]=i
    return n3

     
def iterate(Z):
    R,C = len(Z), len(Z[0])
    n0 = neighbours0(Z)
    n1 = neighbours1(Z)
    n2 = neighbours2(Z)
    n3 = neighbours3(Z)
    for x in range(1,R-1):
        for y in range(1,C-1):
            if Z[x][y]==0 and n1[x][y]>=1:
                Z[x][y]=1
            elif Z[x][y]==1 and n1[x][y]>=2:
                Z[x][y]=2
            elif Z[x][y]==2 and n3[x][y]>=2:
                Z[x][y]=3
            elif Z[x][y]==3 and (n2[x][y]<2 or n3[x][y]<2):
                Z[x][y]=0
            elif Z[x][y]==2 and (n1[x][y]<2 or n2[x][y]<2):
                Z[x][y]=0
            elif Z[x][y]==1 and (n1[x][y]<2 or n2[x][y]>=2):
                Z[x][y]=0
            elif Z[x][y]==2 and (n1[x][y]<2 or n3[x][y]>2):
                Z[x][y]=1                    
            elif Z[x][y]==0 and n2[x][y]>=2 and n3[x][y]<1:
                Z[x][y]=2
            elif Z[x][y]==0 and n3[x][y]>=2:
                Z[x][y]=3    
    return Z

def show(Z):
    for l in Z[1:-1]: 
        print l[1:-1]
    print
show(Z)

disp = matplotlib.pyplot.figure()
graph=disp.add_subplot(111)
disp.show()
E=1
while E>0:
    iterate(Z)
    graph.clear()
    graph.imshow(Z,interpolation='nearest', cmap=matplotlib.pyplot.cm.pink_r)
    matplotlib.pyplot.pause(0.5)
    E=E+1


