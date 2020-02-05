import os

class MBCF(object):
	"""docstring for MBCF"""

	def __init__(self, base_file, test_file):
		super(MBCF, self).__init__()
		self.base_file = base_file
		self.test_file = test_file
		self.ru = {}
		self.ri = {}
		self.MAE_1,self.MAE_2,self.MAE_3 = 0.0,0.0,0.0
		self.RMSE_1,self.RMSE_2,self.RMSE_3 = 0.0,0.0,0.0
		self._UCF = 0.5
		
	def load_data(self):
		for line in open(self.base_file,"r",encoding="utf-8"):
			record = line.split()
			if int(record[0]) not in self.ru.keys():
				self.ru[int(record[0])] = {int(record[1]):float(record[2])}
			else:
				self.ru[int(record[0])][int(record[1])] = float(record[2])

			if int(record[1]) not in self.ri.keys():
				self.ri[int(record[1])] = {int(record[0]):float(record[2])}
			else:
				self.ri[int(record[1])][int(record[0])] = float(record[2])

	def cal_similarity_u(self,u,w):
		_u = sum(self.ru[u].values())/len(self.ru[u])
		_w = sum(self.ru[w].values())/len(self.ru[w])
		numerator, denominator_a, denominator_b = 0.0, 0.0, 0.0
		for key in self.ru[u].keys():
			if key in self.ru[w].keys():
				numerator += (self.ru[u][key]-_u)*(self.ru[w][key]-_w)
				denominator_a += (self.ru[u][key]-_u)**2
				denominator_b += (self.ru[w][key]-_w)**2
		denominator_a = denominator_a**0.5
		denominator_b = denominator_b**0.5
		if denominator_a==0.0 or  denominator_b==0.0:
			return 0.0
		return numerator/(denominator_a*denominator_b)

	def swu_rating(self,u,j):
		numerator, denominator= 0.0, 0.0
		for key in self.ru.keys():
			if key != u:
				similarity = self.cal_similarity_u(key, u)
				if similarity > 0.0 and j in self.ru[key].keys():
					_w = sum(self.ru[key].values())/len(self.ru[key])
					numerator += similarity*(self.ru[key][j]-_w)
					denominator += similarity
		if numerator==0.0 or denominator==0.0:
			return 0
		return sum(self.ru[u].values())/len(self.ru[u])+numerator/denominator

	def cal_similarity_i(self,k,j):
		numerator, denominator_a, denominator_b = 0.0, 0.0, 0.0
		for key in self.ri[k].keys():
			if key in self.ri[j].keys():
				_u = sum(self.ru[key].values())/len(self.ru[key])
				numerator += (self.ri[k][key]-_u)*(self.ri[j][key]-_u)
				denominator_a += (self.ri[k][key]-_u)**2
				denominator_b += (self.ri[j][key]-_u)**2
		denominator_a = denominator_a**0.5
		denominator_b = denominator_b**0.5
		if denominator_a==0.0 or  denominator_b==0.0:
			return 0.0
		return numerator/(denominator_a*denominator_b)

	def skj_rating(self,u,j):
		numerator, denominator= 0.0, 0.0
		for key in self.ri.keys():
			if key != j:
				similarity = self.cal_similarity_i(key, j)
				if similarity > 0.0 and key in self.ru[u].keys():
					numerator += similarity*(self.ru[u][key])
					denominator += similarity
		if numerator==0.0 or denominator==0.0:
			return 0
		return numerator/denominator

	def results(self):
		for line in open(self.test_file,"r",encoding="utf-8"):
			record = line.split()
			if int(record[0]) in self.ru.keys() and int(record[1]) in self.ri.keys():
				_rui_1 = self.swu_rating(int(record[0]), int(record[1]))
				_rui_2 = self.skj_rating(int(record[0]), int(record[1]))
				_rui_3 = self._UCF*_rui_1 + (1-self._UCF)*_rui_2
				if _rui_1 != 0:
					self.MAE_1 += abs(float(record[2]) - _rui_1)/20000
					self.RMSE_1 += (float(record[2]) - _rui_1)**2/20000
				if _rui_2 != 0:
					self.MAE_2 += abs(float(record[2]) - _rui_2)/20000
					self.RMSE_2 += (float(record[2]) - _rui_2)**2/20000
				if _rui_3 != 0:
					self.MAE_3 += abs(float(record[2]) - _rui_3)/20000
					self.RMSE_3 += (float(record[2]) - _rui_3)**2/20000
		self.RMSE_1 = self.RMSE_1**0.5
		self.RMSE_2 = self.RMSE_2**0.5
		self.RMSE_3 = self.RMSE_3**0.5
		print("User-based CF\n",self.RMSE_1,self.MAE_1)
		print("Item-based CF\n",self.RMSE_2,self.MAE_2)
		print("Hybself.rid CF\n",self.RMSE_3,self.MAE_3)

def main():
	# load u1.base and u1.test's data into memory
	base_file = str(os.getcwd()).replace("\\","/") + "/ml-100k/u1.base"
	test_file = str(os.getcwd()).replace("\\","/") + "/ml-100k/u1.test"
	# initial the class
	mbcf = MBCF(base_file, test_file)
	mbcf.load_data()
	mbcf.results()

if __name__ == '__main__':
	main()