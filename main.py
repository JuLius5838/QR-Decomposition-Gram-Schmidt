from TP2 import *
from lib_LU_Chol import *
import time
import numpy as np
import math as m
import scipy.linalg
import matplotlib.pyplot as plt


size_ = [3,4,5,10,15,50,100]
cholesky_total_time = []
cholesky_total_errors = []
gauss_total_time = []
gauss_total_errors = []
QR_total_time = []
QR_total_errors = []


for i in range(len(size_)):

	A = Matrix_A(size_[i])
	B = Matrix_B(size_[i])
	
	start_cholesky = time.time()
	solutions_cholesky = resol_cholesky(A, B)
	end_cholesky = time.time()
	time_cholesky = end_cholesky - start_cholesky
	erreur_cholesky_ = abs(np.dot(A, solutions_cholesky) - B)
	erreur_cholesky = np.average(erreur_cholesky_)


	A_2 = Matrix_A(size_[i])
	B_2 = Matrix_B(size_[i])

	start_gauss = time.time()
	solutions_gauss = Gauss(A_2, B_2)
	end_gauss = time.time()
	time_gauss = end_gauss - start_gauss
	erreur_gauss_ = abs(np.dot(A_2, solutions_gauss) - B_2)
	erreur_gauss = np.average(erreur_gauss_)


	A_3 = Matrix_A(size_[i])
	B_3 = Matrix_B(size_[i])

	start_QR = time.time()
	solutions_QR = ResolGS(A_3, B_3)
	end_QR = time.time()
	time_QR = end_QR - start_QR
	erreur_QR_ = abs(np.dot(A_3, solutions_QR) - B_3)
	erreur_QR = np.average(erreur_QR_)

	cholesky_total_time.append(time_cholesky)
	cholesky_total_errors.append(erreur_cholesky)
	gauss_total_time.append(time_gauss)
	gauss_total_errors.append(erreur_gauss)
	QR_total_time.append(time_QR)
	QR_total_errors.append(erreur_QR)


figure_temps, xy = plt.subplots()

xy.set_xlabel('size')
xy.set_ylabel('time')
xy.plot(size_, cholesky_total_time, label = 'Temps Cholesky', color = 'blue')
xy.plot(size_, gauss_total_time, label = 'Temps Gauss', color = 'blue', linestyle = 'dashed')
xy.plot(size_, QR_total_time, label = 'Temps QR', color = 'blue', linestyle = 'dotted')
plt.legend()
plt.title('Comparaison des temps mis pour Gauss, Cholesky et la decompo QR')
plt.show()

figure_erreurs, wz = plt.subplots()

wz.set_xlabel('size')
wz.set_ylabel('errors')
wz.plot(size_, cholesky_total_errors, label = 'Erreur Cholesky', color = 'red')
wz.plot(size_, gauss_total_errors, label = 'Erreur Gauss', color = 'red', linestyle = 'dashed')
wz.plot(size_, QR_total_errors, label = 'Erreur QR', color = 'red', linestyle = 'dotted')
plt.legend()
plt.title('Comparaison des erreurs pour Gauss, Cholesky et la decompo QR')
plt.show()
