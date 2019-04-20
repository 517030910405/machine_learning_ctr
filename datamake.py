RLi = ["label,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,F16,F17,F18,F19,F20,F21,F22,F23,F24"]



fo = open("train.data","r")
Li = fo.read().split("\n")
Li.pop()
cnt=10
print(len(Li))
for item in Li:
	cnt -= 1
	if True :
		LL = item.split(" ")
		LLL = []
		for item2 in LL:
			LLL.append(item2.split(":")[0])
		RLi.append(",".join(LLL))
RLi.append("")

fo = open("test.data","r")
Li = fo.read().split("\n")
Li.pop()
cnt=10
print(len(Li))
for item in Li:
	cnt -= 1
	if True :
		LL = item.split(" ")
		LLL = []
		for item2 in LL:
			LLL.append(item2.split(":")[0])
		RLi.append(",".join(LLL))
RLi.append("")




oopoop = open("super_train.csv","w")
oopoop.write("\n".join(RLi))