import sys
import os
import time
import json
import random
# import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import export_graphviz
from graphviz import Source


HEAVY_HITTER_THRESH = 0.05

# def plot_ports_freq(counts, name):
# 	ports_freq = list(counts.items())
# 	sorted_ports_freq = sorted(ports_freq, key=lambda x: x[1])
# 	ports, freq = zip(*sorted_ports_freq)
# 	fig = plt.figure(figsize = (20,10))
# 	plt.bar(ports, freq)
# 	plt.xlabel("Output ports")
# 	plt.ylabel("Number of mapped entries")
# 	plt.title("Frequency for %s" %(name))
# 	plt.xticks([])
# 	# plt.show()
# 	plt.savefig("./plots/%s_port_freq.png" %(name))
# 	print(name, freq[-5:])

def heavy_hitters(counts):
	ports, freq = zip(*list(counts.items()))
	max_count = max(freq)
	heavy_hitters_ = []
	for p, f in counts.items():
		if f > HEAVY_HITTER_THRESH*max_count:
			heavy_hitters_.append(p)
	return heavy_hitters_

def classify_prefixes_by_hh(prefix_port, heavy_hits):
	xs, ys = zip(*[(prefix, 1) if port in heavy_hits else (prefix, 0) for prefix, port in prefix_port])
	return xs, ys

def find_heavy_hitters(cr):
	counts_at_ports = {}
	fw_table = cr[1]['fw_table']
	prefix_port_tup = [(x['prefix'], x["action"]["py/tuple"][0]) for x in fw_table]
	prefix_port_tup = [(x,y) for x,y in prefix_port_tup if '.' in y]
	for _,port in prefix_port_tup:
		if port not in counts_at_ports:
			counts_at_ports[port] = 0
		counts_at_ports[port] += 1
	# plot_ports_freq(counts_at_ports, cr[1]['name'])
	heavy_hits = heavy_hitters(counts_at_ports)
	x, y = classify_prefixes_by_hh(prefix_port_tup, heavy_hits)
	print("%s: %d heavy hitter counts, %d non-heavy hitters count" %(cr[1]['name'], sum(y), len(y)-sum(y)))
	return x, y

def prefix_features(ip_prefix):
	ip, prefix = ip_prefix.replace('*', '').split('/')
	return [int(x) for x in ip.split('.')] + [int(prefix)]

feature_names = ['IP-1', 'IP-2', 'IP-3', 'IP-4', 'Prefix Len']

def check_model(m, ms, xtr, ytr, xte, yte):
	t1 = time.time()
	m.fit(xtr, ytr)
	t2 = time.time()
	mpred = m.predict(xte)
	print(accuracy_score(yte, mpred), " Accuracy score FOR model ", ms)
	conf_mat = confusion_matrix(y_test, mpred)
	print(conf_mat.shape, " FOR model: ", ms, " TrainTime: ", t2-t1)
	dt = m
	if "RandomForest" in ms:
		dt = m.estimators_[0]
	export_graphviz(dt, out_file=ms+'.dot', 
                feature_names = feature_names,
                rounded = True, proportion = False, 
                precision = 2, filled = True)
	s = Source.from_file(ms+'.dot')
	s.view()

total_routers = 0
core_routers = []
bd_routers = []
others = []
fname = sys.argv[1] #"../parsed/cisco"
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

print(len(core_routers), len(bd_routers), len(others))

core_routers = sorted( core_routers, key=lambda x: len(x[1]["fw_table"]) )

for cr in core_routers:
	if cr[1]['name'] == 'core4-1':
		xs,ys = find_heavy_hitters(cr)
		xs = [prefix_features(x) for x in xs]
		x_train, x_test, y_train, y_test = train_test_split(xs,ys,train_size=int(len(ys)*0.8), random_state=21)

		dt_model = DecisionTreeClassifier(max_depth=4)
		check_model(dt_model, cr[1]['name']+"HHDecisionTree", x_train, y_train, x_test, y_test)

		rfc_model = RandomForestClassifier(n_estimators=10, max_depth=4) # TODO2: add tree depth parameter
		check_model(rfc_model, cr[1]['name']+"HHRandomForest", x_train, y_train, x_test, y_test)

		# ada_model = AdaBoostClassifier(
		#     DecisionTreeClassifier(max_depth=5), n_estimators=100)
		# check_model(ada_model, "ADA + DT", x_train, y_train, x_test, y_test)



# core9-2 (21, 48, 106, 111, 2994)
# core8-3 (1, 1, 1, 1, 3376)
# core4-3 (1, 1, 1, 1, 3375)
# core6-1 (1, 3364)
# core9-1 (32, 48, 85, 1394, 2395)
# core8-1 (300, 702, 833, 945, 979)
# core2-2 (210, 371, 663, 707, 1006)
# core8-2 (384, 701, 795, 1121, 1122)
# core1-2 (197, 684, 699, 727, 819)
# core1-1 (171, 370, 682, 863, 1100)
# core4-2 (171, 738, 804, 878, 1233)
# core2-1 (41, 114, 170, 375, 2093)
# core10-1 (5, 6, 3346, 3346)
# core5-1 (27, 35, 35, 961, 2021)
# core4-1 (161, 505, 561, 901, 1337)

# core9-2: 2994 heavy hitter counts, 394 non-heavy hitters count
# core8-3: 3376 heavy hitter counts, 205 non-heavy hitters count
# core4-3: 3375 heavy hitter counts, 205 non-heavy hitters count
# core6-1: 3364 heavy hitter counts, 1 non-heavy hitters count
# core9-1: 3789 heavy hitter counts, 362 non-heavy hitters count
# core8-1: 3759 heavy hitter counts, 375 non-heavy hitters count
# core2-2: 2957 heavy hitter counts, 1056 non-heavy hitters count
# core8-2: 4123 heavy hitter counts, 380 non-heavy hitters count
# core1-2: 3565 heavy hitter counts, 1194 non-heavy hitters count
# core1-1: 3334 heavy hitter counts, 1336 non-heavy hitters count
# core4-2: 3824 heavy hitter counts, 2046 non-heavy hitters count
# core2-1: 2752 heavy hitter counts, 3116 non-heavy hitters count
# core10-1: 6692 heavy hitter counts, 11 non-heavy hitters count
# core5-1: 2982 heavy hitter counts, 3591 non-heavy hitters count
# core4-1: 3465 heavy hitter counts, 4037 non-heavy hitters count
