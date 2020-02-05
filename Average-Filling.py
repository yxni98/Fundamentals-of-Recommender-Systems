#!user/bin/env python3  
# -*- coding: utf-8 -*-

import os

class AF:
	"""docstring for AF"""

	# some variables to be used
	def __init__(self, base_file, test_file):
		super(AF, self).__init__()
		self.base_file = base_file
		self.test_file = test_file
		self.record_count = 0
		self.r = 0.0
		self.ru = {}
		self.ri = {}
		self.u_i = {}
		self.i_u = {}
		self.bu = {}
		self.bi = {}
		self.MAE = 0.0
		self.RMSE = 0.0

	# get the r, ru, ri, bu, bi of the base data
	def r_ru_ri_bu_bi(self):	
		# line data: user id | item id | rating | timestamp
		for line in open(self.base_file,"r",encoding="utf-8"):
			record = line.split()
			self.record_count += 1
			self.r += float(record[2])
			if int(record[0]) not in self.ru.keys():
				self.u_i[int(record[0])] = [int(record[1])]
				self.ru[int(record[0])] = [float(record[2]),1]
			else:
				temp_sum = self.ru[int(record[0])][0] + float(record[2])
				temp_count = self.ru[int(record[0])][1] + 1
				self.ru[int(record[0])][0] = temp_sum
				self.ru[int(record[0])][1] = temp_count
				temp_list = self.u_i[int(record[0])]
				temp_list.append(int(record[1]))
				self.u_i[int(record[0])] = temp_list

			if int(record[1]) not in self.ri.keys():
				self.i_u[int(record[1])] = [int(record[0])]
				self.ri[int(record[1])] = [float(record[2]),1]
			else:
				temp_sum = self.ri[int(record[1])][0] + float(record[2])
				temp_count = self.ri[int(record[1])][1] + 1
				self.ri[int(record[1])][0] = temp_sum
				self.ri[int(record[1])][1] = temp_count
				temp_list = self.i_u[int(record[1])]
				temp_list.append(int(record[0]))
				self.i_u[int(record[1])] = temp_list
		for key in self.ru.keys():
			self.ru[key] = [self.ru[key][0],self.ru[key][1],self.ru[key][0]/self.ru[key][1]]
		for key in self.ri.keys():
			self.ri[key] = [self.ri[key][0],self.ri[key][1],self.ri[key][0]/self.ri[key][1]]
		self.r /= self.record_count

		for key in self.u_i.keys():
			r = 0.0
			for i in self.u_i[key]:
				r += self.ri[i][2]
			self.u_i[key] = r
		for key in self.i_u.keys():
			r = 0.0
			for i in self.i_u[key]:
				r += self.ru[i][2]
			self.i_u[key] = r
		for key in self.ru.keys():
			self.bu[key] = (self.ru[key][0] - self.u_i[key])/self.ru[key][1]
			self.ru[key] = self.ru[key][2]
		for key in self.ri.keys():
			self.bi[key] = (self.ri[key][0] - self.i_u[key])/self.ri[key][1]
			self.ri[key] = self.ri[key][2]

		return self.r, self.ru, self.ri, self.bu, self.bi

	# finish the first and second scan of the base data(include ru,ri,bu,bi)
	def scan(self,N,M,r,ru,ri,bu,bi):
		n,m = N,M
		for i in range(1,m+1):
			if i <= n and i not in ru.keys():
				ru[i] = r
			if i <= m and i not in ri.keys():
				ri[i] = r
			if i <= n and i not in bu.keys():
				bu[i] = 0
			if i <= m and i not in bi.keys():
				bi[i] = 0
		return r, ru, ri, bu, bi

	# get the result(MAE,RMSE)
	def MAE_RMSE(self, r, ru, ri, bu, bi, rule):
		self.MAE, self.RMSE = 0.0, 0.0
		for line in open(self.test_file,"r",encoding="utf-8"):
			record = line.split()
			if rule == "rui=ru":
				rui = ru[int(record[0])]
			elif rule == "rui=ri":
				rui = ri[int(record[1])]
			elif rule == "rui=ru/2+ri/2":
				rui = ru[int(record[0])]/2.0 + ri[int(record[1])]/2.0
			elif rule == "rui=bu+ri":
				rui = bu[int(record[0])] + ri[int(record[1])]
			elif rule == "rui=ru+bi":
				rui = ru[int(record[0])] + bi[int(record[1])]
			elif rule == "rui=r+bu+bi":
				rui = r + bu[int(record[0])] + bi[int(record[1])]
			self.MAE += abs(float(record[2]) - rui)/20000
			self.RMSE += ((float(record[2]) - rui)**2)/20000
		self.RMSE = self.RMSE**0.5
		print(rule)
		print("RMSE：" + str(self.RMSE), "MAE：" + str(self.MAE))

def main():
	# load u1.base and u1.test's data into memory
	base_file = str(os.getcwd()).replace("\\","/") + "/ml-100k/u1.base"
	test_file = str(os.getcwd()).replace("\\","/") + "/ml-100k/u1.test"
	# initial the class
	af = AF(base_file, test_file)
	# call the function to get r,ru,ri,bu,bi
	r,ru,ri,bu,bi = af.r_ru_ri_bu_bi()
	# call the function to scan ru,ri,bu,bi
	r,ru,ri,bu,bi = af.scan(943,1682,r,ru,ri,bu,bi)
	# predict rules
	rules = ["rui=ru","rui=ri","rui=ru/2+ri/2","rui=bu+ri","rui=ru+bi","rui=r+bu+bi"]
	for rule in rules:
		af.MAE_RMSE(r,ru,ri,bu,bi,rule)

if __name__ == '__main__':
	main()