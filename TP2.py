import numpy as np
from math import *
import scipy.linalg
import random
import matplotlib.pyplot as plt
import time

def DecompositionGS (A):
	n,m = np.shape(A)
	Q = np.zeros([n,n])
	R = np.zeros([n,n])

	for j in range (0,n):
		for i in range (0,j):
			if (i<j):
				R[i,j]=np.dot(A[:,j],Q[:,i])
		S=0
		for k in range(0, j):
			S = S + R[k,j]*Q[:,k]
		w = A[:,j] - S
		R[j,j] = np.linalg.norm(w)

		Q[:,j] = (1/R[j,j])*w

	return(Q,R)

def ResolGS(A,b):
	Q,R = DecompositionGS(A)
	C = np.dot(Q.T,b)
	n =  len(b)
	solutions= np.shape(b)
	x = np.zeros(solutions)
	
	for j in range(len(A)-1,-1,-1):
		S=0
		for k in range(j+1,n):
			S = S + R[j,k]*x[k]
		x[j] = (C[j]-S)/R[j,j]
	return x 
 
	