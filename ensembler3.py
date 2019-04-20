import numpy as np
class ensembler:
	def __init__(self):
		self.arr = None
		self.weights = 0.0
		assert True
	def ensemble(self,str,weight):
		if (self.weights == 0):
			so = np.loadtxt(str,dtype = np.float64)
			print(so.shape)
			if (len(so.shape)==2):
				so = so[:,1:2]
			so = so.reshape((-1,))
			so = np.log(1/so-1)
			
			#almost_yes = (so-2)*np.greater(so-2,0)
			#almost_no = (so+2)*np.greater(0,so+2)
			#so += almost_no
			#so += almost_yes
			#so = so*np.abs(so)
			
			self.arr = so * weight
			self.weights +=	 weight
		else:
			so = np.loadtxt(str,dtype = np.float64)
			print(so.shape)
			if (len(so.shape)==2):
				so = so[:,1:2]
			so = so.reshape((-1,))
			so = np.log(1/so-1)
			
			#almost_yes = (so-2)*np.greater(so-2,0)
			#almost_no = (so+2)*np.greater(0,so+2)
			#so += almost_no
			#so += almost_yes
			#so = so*np.abs(so)
			
			
			self.arr += so * weight
			self.weights += weight
	def savetxt(self,str):
		np.savetxt(str , (1/(np.exp(self.arr / self.weights)+1)))

	def savecsv(self,Str):
		ans = (1/(np.exp(self.arr / self.weights)+1))
		#ans = ((self.arr / self.weights))
		myl = []
		myl.append("id,Predicted")
		for i in range(1000000):
			myl.append(str(i)+","+str(ans[i]))
		fo = open(Str,"w")
		fo.write("\n".join(myl))
if (__name__ == "__main__"):
	print("dodo")
	ens = ensembler()
	'''
	#ens.ensemble("nffm1.ans1",1)
	#ens.ensemble("nffm2.ans1",1)
	#ens.ensemble("nffm3.ans1",1)
	#ens.ensemble("nffm4_7622.ans1",2)
	ens.ensemble("nffm5_7597.ans1",2)
	ens.ensemble("nffm6_7604.ans1",2)
	ens.ensemble("nffm7_7595.ans1",2)
	ens.ensemble("nffm8_7613.ans1",2)
	ens.ensemble("nffm9_7598.ans1",2)
	#ens.ensemble("nffm10_7577.ans1",2)
	ens.ensemble("nffm11_7606.ans1",2)
	'''
	
	ens.ensemble("nffm12_7654.ans1",2)
	ens.ensemble("nffm13_7632.ans1",2)
	ens.ensemble("nffm14_7648.ans1",2)
	ens.ensemble("nffm15_7645.ans1",2)
	ens.ensemble("nffm16_7649.ans1",2)
	
	ens.savecsv("new_idea7_log_mean.csv")