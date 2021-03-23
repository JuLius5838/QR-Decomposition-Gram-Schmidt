import numpy as np
from math import sqrt


#reduction_Gauss

def Matrix_A(size):			# creer une matrice A definie positive

	M = np.random.randint(-10, +10, size = (size, size))
	M = np.array(M,float)
	A = np.dot(M.T, M)
	return A


def Matrix_B(size):			# creer une matrice colonne B de la meme taille que A

	B = np.random.randint(-10, 10, size = (size, 1))
	B = np.array(B,float)
	return B




def transposed_matrix(matrix):		#transpose une matrice (pas utile)
	matrix_t = matrix.transpose()
	return matrix_t



def cholesky(matrix):

	size = np.shape(matrix)
	L = np.zeros(size)

	for j in range(size[1]):
		L[j][j] = sqrt(matrix[j][j] - sum([L[j][k]**2 for k in range(j)]))

		for i in range(j + 1, size[0]):
			L[i][j] = (matrix[i][j] - sum([L[i][k] * L[j][k] for k in range(j)])) / L[j][j]

	return L




def resol_cholesky(matrix_A, matrix_B):

	solutions = np.shape(matrix_B)
	L = cholesky(matrix_A)
	Y = np.zeros(solutions)

	for i in range(solutions[0]):
		somme = [L[i][k] * Y[k][0] for k in range(i)]
		Y[i][0] = (matrix_B[i][0] - sum(somme)) / L[i][i]

	LT = transposed_matrix(L)
	X = np.zeros(solutions)


	for i in range(solutions[0] - 1, -1, -1):
		somme = [LT[i][k] * X[k][0] for k in range(solutions[0] - 1, i, -1)]
		X[i][0] = (Y[i][0] - sum(somme)) / LT[i][i]

	return X


def reduction_Gauss(Aaug):


	n = len(Aaug)
	for k in range(n):

		if Aaug[k][k] == 0:
			print("Error: ne peut pas etre resolu (pivot = 0)")
			stop = 1
			return
		pivot = Aaug[k] / Aaug[k][k]
		for i in range(k+1, n):
			Aaug[i] = Aaug[i] - pivot * Aaug[i][k]
	#print('Matrice augmentee:')
	#print(Aaug)
	#print('\n')

def ResolutionSystTriSup(Aaug):

	n = len(Aaug)
	solutions = np.array([[0]] * n, dtype = float)
	
	for i in range(n - 1, -1, -1):
		ligne = Aaug[i]
		x = 0

		for j in range(i + 1, n):
			x = x + solutions[j] * ligne[j]
		solutions[i][0] = (ligne[-1] - x) / ligne[i] 

	return solutions


def Gauss(A, B):
	
	Aaug = np.concatenate((A, B), axis=1)
	#print(Aaug)
	reduction_Gauss(Aaug)
	x = ResolutionSystTriSup(Aaug)
	return x