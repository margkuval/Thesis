import numpy as np
import matplotlib.pyplot as plt

"""Members characteristics"""
iEdge = np.array([0, 1, 2])  #beginning of an edge
jEdge = np.array([1, 2, 0])  #end of an edge
xcoord = np.array([0, 3, 1])
ycoord = np.array([0, 0, 5])

numnode = xcoord.shape[0]  #all nodes must be used
numelem = iEdge.shape[0]  #count # of beginnings

u = np.vstack([[2*iEdge, 2*iEdge+1], [2*jEdge, 2*jEdge+1]]).transpose()

print(u)

#jak ma vedet, ze node 0 ma posuny 0,1

k = np.ix_(iEdge, jEdge)

"""Material characteristics"""
E = np.array([40000, 40000, 40000])  #modulus of elasticity for each member
A = np.array([0.01, .02, 0.02])  #area - each member

"Connecting MAT computation"
ij = np.array([iEdge, jEdge])



"Basic stiffness MAT"

