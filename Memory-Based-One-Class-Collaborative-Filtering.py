#!user/bin/env python3  
# -*- coding: utf-8 -*-

import os

class MBOCCF(object):
	"""docstring for MBOCCF"""
	def __init__(self):
		super(MBOCCF, self).__init__()
		self.k = 5
		self.K = 50
		self.n = 943
		self.m = 1682
		self.Iu = {}
		self.bi = {}
		self.ite = {}
		self.u_i_k = {}
		self.Ui = {}
		self.all_items = set()
		self.all_users = set()
		self.top_k_items = {}
		self.top_k_users = {}

	def pre_processing(self):
		for line in open(str(os.getcwd()).replace("\\","/") + "/ml-100k/u1.base.occf","r",encoding="utf-8"):
			record = line.split()
			self.all_items.add(int(record[1]))
			self.all_users.add(int(record[0]))
			if int(record[0]) not in self.Iu.keys():
				self.Iu[int(record[0])] = [int(record[1])]
			else:
				temp_list = self.Iu[int(record[0])]
				temp_list.append(int(record[1]))
				self.Iu[int(record[0])] = temp_list # 每个用户及被打了分的物品映射 u1->[i1,i2...],u2->[i1,i2...] ...

			if int(record[1]) not in self.Ui.keys():
				self.Ui[int(record[1])] = [int(record[0])]
			else:
				temp_list = self.Ui[int(record[1])]
				temp_list.append(int(record[0]))
				self.Ui[int(record[1])] = temp_list # 每个物品及被打了分的用户映射 i1->[u1,u2...],i2->[u1,u2...] ...

		for line in open(str(os.getcwd()).replace("\\","/") + "/ml-100k/u1.test.occf","r",encoding="utf-8"):
			record = line.split()
			if int(record[0]) not in self.ite.keys():
				self.ite[int(record[0])] = [int(record[1])]
			else:
				temp_list = self.ite[int(record[0])]
				temp_list.append(int(record[1]))
				self.ite[int(record[0])] = temp_list # 每个用户在test中对应的物品映射 u1->[i1,i2...],u2->[i1,i2...] ...

	def PopRank(self):
		self.u_i_k = {}
		u = 44140/self.n/self.m
		for line in open(str(os.getcwd()).replace("\\","/") + "/ml-100k/u1.base.occf","r",encoding="utf-8"):
			record = line.split()
			if int(record[1]) not in self.bi.keys():
				self.bi[int(record[1])] = 1
			else:
				temp = self.bi[int(record[1])]
				temp += 1
				self.bi[int(record[1])] = temp
		for key in self.bi.keys():
			self.bi[key] = self.bi[key]/self.n - u
		dict_list = sorted(self.bi.items(),key = lambda x:x[1],reverse = True) # 预测的多项物品的ranking（降序排列）
			
		for ui_key in self.Iu.keys():
			count = 0
			for data in dict_list:
				if data[0] not in self.Iu[ui_key]:
					if count == 0:
						self.u_i_k[ui_key] = [data[0]]
					else:
						temp_list = self.u_i_k[ui_key]
						temp_list.append(data[0])
						self.u_i_k[ui_key] = temp_list # 每个用户的前n个没推荐过的物品id（降序排列）
					count += 1

	def Item_Jaccard_Index(self, k, j): # 计算物品的相似度： 交集 除以 并集
		if k in self.Ui.keys() and j in self.Ui.keys():
			numerator = len(set(self.Ui[k]).intersection(set(self.Ui[j])))
			if numerator == 0:
				return 0
			else:
				denominator = len(set(self.Ui[k]).union(set(self.Ui[j])))
				return numerator/denominator
		else:
			return 0	

	def User_Jaccard_Index(self, k, j): # 计算用户的相似度： 交集 除以 并集
		if k in self.Iu.keys() and j in self.Iu.keys():
			numerator = len(set(self.Iu[k]).intersection(set(self.Iu[j])))
			if numerator == 0:
				return 0
			else:
				denominator = len(set(self.Iu[k]).union(set(self.Iu[j])))
				return numerator/denominator
		else:
			return 0	

	def User_based(self):
		self.u_i_k = {}
		mapping = {}
		for user in self.all_users:
			mapping[user] = {}
			for near_user in self.all_users:
				if user != near_user:
					mapping[user][near_user] = self.User_Jaccard_Index(user, near_user) # 获得用户间的相似度矩阵

		for user in self.all_users: # 根据每行对应的value值选取出top-k
			self.top_k_users[user] = [x[0] for x in (sorted(mapping[user].items(), key = lambda x:x[1], reverse = True)[:self.K])]

		for key in self.ite.keys():
			self.u_i_k[key] = {}
			for item in self.all_items:
				if item not in self.Iu[key]: # 已经打过高分的不纳入推荐范围，得到self.u_i_k
					user_list = list(set(self.Ui[item]).intersection(set(self.top_k_users[key])))
					ruj = 0.0
					for w in user_list:
						ruj += mapping[w][key]
					self.u_i_k[key][item] = ruj
		
		return self.u_i_k

	def Item_based(self):
		self.u_i_k = {}
		mapping = {}
		for item in self.all_items:
			mapping[item] = {}
			for near_item in self.all_items:
				if item != near_item:
					mapping[item][near_item] = self.Item_Jaccard_Index(item, near_item) # 获得物品间的相似度矩阵

		for item in self.all_items: # 根据每行对应的value值选取出top-k
			self.top_k_items[item] = [x[0] for x in (sorted(mapping[item].items(), key = lambda x:x[1], reverse = True)[:self.K])]

		for key in self.ite.keys():
			self.u_i_k[key] = {}
			for item in self.all_items:
				if item not in self.Iu[key]: # 已经打过高分的不纳入推荐范围，得到self.u_i_k
					item_list = list(set(self.Iu[key]).intersection(set(self.top_k_items[item])))
					ruj = 0.0
					for k in item_list:
						ruj += mapping[k][item]
					self.u_i_k[key][item] = ruj

		return self.u_i_k

	def recommend_list(self):
		for key in self.u_i_k.keys(): # 对self.u_i_k进行排序
			self.u_i_k[key] = [x[0] for x in sorted(self.u_i_k[key].items(), key = lambda x:x[1], reverse = True)]

	def Hybrid_based(self):
		a = 0.4
		user_u_i_k = self.User_based()
		item_u_i_k = self.Item_based() # 获取两个u_i_k，加入参数得到混合的u对i的打分
		self.u_i_k = {}
		for user in user_u_i_k.keys():
			self.u_i_k[user] = {}
			for item in user_u_i_k[user].keys():
				ruj = a*user_u_i_k[user][item] + (1-a)*item_u_i_k[user][item]
				self.u_i_k[user][item] = ruj
	
		for key in self.u_i_k.keys(): # 对self.u_i_k进行排序
			self.u_i_k[key] = [x[0] for x in sorted(self.u_i_k[key].items(), key = lambda x:x[1], reverse = True)]

	def Pre_k(self): # 找出re前k个item中与test的交集个数，除以k，最后除以test用户数
		Pre_k = 0.0
		for key in self.ite.keys():
			Pre_k += (len(set(self.ite[key]).intersection(set(self.u_i_k[key][:self.k]))) / (self.k*len(self.ite.keys())))
		print("Pre©5: "+str(Pre_k))

	def Rec_k(self): # 找出re前k个item中与test的交集个数，除以用户对应的test中的item个数，最后除以test用户数
		Rec_k = 0.0
		for key in self.ite.keys():
			Rec_k += (len(set(self.ite[key]).intersection(set(self.u_i_k[key][:self.k]))) / (len(self.ite[key])*len(self.ite.keys())))
		print("Rec©5: "+str(Rec_k))

def main():
	mboccf = MBOCCF()
	mboccf.pre_processing()
	print("PopRank:")
	mboccf.PopRank()
	mboccf.Pre_k()
	mboccf.Rec_k()
	print("Item_based:")
	mboccf.Item_based()
	mboccf.recommend_list()
	mboccf.Pre_k()
	mboccf.Rec_k()
	print("User_based:")
	mboccf.User_based()
	mboccf.recommend_list()
	mboccf.Pre_k()
	mboccf.Rec_k()
	print("Hybrid_based:")
	mboccf.Hybrid_based()
	mboccf.Pre_k()
	mboccf.Rec_k()

if __name__ == '__main__':
	main()