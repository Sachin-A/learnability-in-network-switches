import sys
import os
import time
import json
import random
import matplotlib.pyplot as plt
	

def convert_ip_to_int(ips):
	ipsplit = ips.replace('*','').split('.')
	ipx = 0
	if len(ipsplit) != 4:
		return 0
	for i in range(4):
		ipx += (int(ipsplit[i]) * pow(2,8*(3-i)))
	return ipx

def plot_sorted_fanout(prefixes, outports, name):
	# prefixes = [convert_ip_to_int(x.split('/')[0]) for x in prefixes]
	prefix_port_zip = list(zip(prefixes, outports))
	prefix_port_zip = [(x,y) for (x,y) in prefix_port_zip if '.' in y]
	sorted_prefix_port = sorted(prefix_port_zip, key=lambda x:x[0])
	unique_ports = list(set([x for _,x in sorted_prefix_port]))
	plottable_prefix_port = [(x,unique_ports.index(y)+1) for (x,y) in sorted_prefix_port]
	plt_prefixes, plt_ports = zip(*plottable_prefix_port)
	fig = plt.figure(figsize = (20,10))
	plt.bar(plt_prefixes, plt_ports)
	plt.xlabel("Sorted Prefixes (converted to int)")
	plt.ylabel("Port numbers")
	plt.title("Pattern for %s" %(name))
	plt.xticks([])
	# plt.show()
	plt.savefig("./plots/%s.png" %(name))

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

core_routers = sorted( core_routers, key=lambda x: len(x[1]["fw_table"]) )

# fw_table = core_routers[-1][1]["fw_table"] #array of entries.

# filter out entries for which action is not a next hop
for c in core_routers:
	fw_table = c[1]['fw_table']
	outports = [ x["action"]["py/tuple"][0] for x in fw_table ] # contains all possible next hop ips, and drop, etc.
	prefixes = [x['prefix'] for x in fw_table]

	print("%s router: %d FIB entries, %d output ports, %d unique prefixes" %(c[1]['name'], len(outports), len(list(set(outports))), len(list(set(prefixes)))))
	plot_sorted_fanout(prefixes, outports, c[1]['name'])


# Found file  core8-1.json #entries:  4331
# Found file  core6-1.json #entries:  3684
# Found file  core5-1.json #entries:  7073
# Found file  core1-1.json #entries:  5044
# Found file  core4-2.json #entries:  6192
# Found file  core4-3.json #entries:  3617
# Found file  core10-1.json #entries:  6958
# Found file  core2-2.json #entries:  4335
# Found file  core8-2.json #entries:  4680
# Found file  core1-2.json #entries:  5027
# Found file  core8-3.json #entries:  3614
# Found file  core2-1.json #entries:  6392
# Found file  core9-1.json #entries:  4280
# Found file  core9-2.json #entries:  3483
# Found file  core4-1.json #entries:  7929
# 15 121 12
# core9-2 router: 3483 FIB entries, 32 output ports, 3483 unique prefixes
# core8-3 router: 3614 FIB entries, 209 output ports, 3614 unique prefixes
# core4-3 router: 3617 FIB entries, 209 output ports, 3617 unique prefixes
# core6-1 router: 3684 FIB entries, 6 output ports, 3684 unique prefixes
# core9-1 router: 4280 FIB entries, 104 output ports, 3586 unique prefixes
# core8-1 router: 4331 FIB entries, 191 output ports, 3718 unique prefixes
# core2-2 router: 4335 FIB entries, 651 output ports, 4272 unique prefixes
# core8-2 router: 4680 FIB entries, 151 output ports, 3663 unique prefixes
# core1-2 router: 5027 FIB entries, 860 output ports, 4446 unique prefixes
# core1-1 router: 5044 FIB entries, 872 output ports, 4523 unique prefixes
# core4-2 router: 6192 FIB entries, 1644 output ports, 5263 unique prefixes
# core2-1 router: 6392 FIB entries, 2626 output ports, 6391 unique prefixes
# core10-1 router: 6958 FIB entries, 8 output ports, 3612 unique prefixes
# core5-1 router: 7073 FIB entries, 3185 output ports, 6939 unique prefixes
# core4-1 router: 7929 FIB entries, 3595 output ports, 7294 unique prefixes