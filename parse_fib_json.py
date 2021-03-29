import sys
import os
import time
import json
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

total_routers = 0
core_routers = []
bd_routers = []
others = []
fname = "../parsed/cisco"
for f in os.listdir(fname):
	if "json" in f:
		total_routers += 1
		if "core" in f:
			with open(fname+"/"+f, 'r') as readf:
				core_routers.append( (f, json.loads(readf.read())) )
				print("Found file ", f, "#entries: ", len(core_routers[-1][1]["fw_table"]) )
		elif "bd" in f:
			bd_routers.append(f)
		else:
			others.append(f)

print(len(core_routers), len(bd_routers), len(others) )

def convert_ip_to_int(ips):
	ipsplit = ips.split('.')
	ipx = 0
	if len(ipsplit) != 4:
		return 0
	for i in range(4):
		ipx += (int(ipsplit[i]) * pow(2,8*(3-i)))
	return ipx

def split_ip_to_ints(ips):
	return [int(x) for x in ips.split('.')]

# random.shuffle(core_routers)
# getting the router with max #entries

core_routers = sorted( core_routers, key=lambda x: len(x[1]["fw_table"]) )
print(core_routers[-2][0])

fw_table = core_routers[1][1]["fw_table"] #array of entries.

# filter out entries for which action is not a next hop
possible_outputs = [ x["action"]["py/tuple"][0] for x in fw_table ] # contains all possible next hop ips, and drop, etc.
output_nhop_input_distr = {}
output_nhop_ids = {}
output_str_input_distr = {}

# distribution of prefixes wrt output [i.e. next hop / attach / recv / drop]
for x in possible_outputs:
	if "." in x:
		if x not in output_nhop_input_distr:
			output_nhop_input_distr[x] = 0
			output_nhop_ids[x] = len(output_nhop_ids)
			if output_nhop_ids[x] % 50 == 7:
				print("Output nhop ip : ", x, " id: ", output_nhop_ids[x])
		output_nhop_input_distr[x] += 1
	else:
		if x not in output_str_input_distr:
			output_str_input_distr[x] = 0
		output_str_input_distr[x] += 1

print(output_str_input_distr)
for k in output_nhop_input_distr:
	if output_nhop_input_distr[k] > 1:
		print(k, output_nhop_input_distr[k])

possible_outputs_str = list(filter(lambda x: "." not in x, possible_outputs))
possible_outputs_nhop = list(filter(lambda x: "." in x, possible_outputs))

possible_outputs_str = set(possible_outputs_str)
possible_outputs_nhop = set(possible_outputs_nhop)

print(possible_outputs_str, len(possible_outputs_nhop), len(output_nhop_ids))

# features:
# split each ip prefix into 4 values + whole ip + prefix length
# arr of arrs
x = []
y = []

for elem in fw_table:
	yi = elem["action"]["py/tuple"][0]
	# ignoring attach/recv/drop AND ip prefixes with *
	if "." in yi and "*" not in elem["prefix"]:
		xi = []
		prefix = elem["prefix"]
		ip = prefix.split('/')[0]
		# prefix length
		xi.append(int(prefix.split('/')[1]))
		# 4-split of ip
		xi += split_ip_to_ints(ip)
		# ip to 32-bit integer
		# xi.append(convert_ip_to_int(ip))

		if convert_ip_to_int(ip) > 0:
			x.append(xi)
			yval = convert_ip_to_int(yi) #output_nhop_ids[yi]
			y.append(yval)
			if yval % 50 == 7:
				print(elem, x[-1], y[-1])

# print("TEST: ", convert_ip_to_int("2.5.3.4"))
# split data into train, test
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=2500, random_state=21)

print(len(x_train), len(x_test), len(y_train), len(y_test))
print(x_train[0], y_train[0], " FIRST elem of train data")

def check_model(m, ms, xtr, ytr, xte, yte):
	t1 = time.time()
	m.fit(xtr, ytr)
	t2 = time.time()
	mpred = m.predict(xte)
	print(accuracy_score(yte, mpred), " Accuracy score FOR model ", ms)
	conf_mat = confusion_matrix(y_test, mpred)
	print(conf_mat.shape, " FOR model: ", ms, " TrainTime: ", t2-t1)

# TODO1: understand confusion matrix
dt_model = DecisionTreeClassifier()
check_model(dt_model, "Decision Tree", x_train, y_train, x_test, y_test)

rfc_model = RandomForestClassifier(n_estimators=10) # TODO2: add tree depth parameter
check_model(rfc_model, "Random Forest", x_train, y_train, x_test, y_test)

ada_model = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=5), n_estimators=100)
check_model(ada_model, "ADA + DT", x_train, y_train, x_test, y_test)

# svm_model = svm.SVC(gamma=0.001)
# check_model(svm_model, "SVM?", x_train, y_train, x_test, y_test)
# lr_model = LogisticRegression()
# check_model(lr_model, "Regression", x_train, y_train, x_test, y_test)
# print(lr_model)