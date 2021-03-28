# Learnability in Network Switches

With the rapidly growing number of hosts connected to the internet, there is a growing demand for fast and in-expensive switch memory. At the same time, the number of network functions handled at the switch are also increasing (e.g., for purposes of routing, telemetry, load balancing etc.) each of which requires dedicated memory. To address these needs, various compact and efficient data structures have been proposed and deployed in the data plane to achieve lower memory costs. Bloom filters, Count Min sketches, and Cuckoo filters are some examples of such data structures found in network switches. But these data structures are probabilistic in nature and hence require an accuracy-memory tradeoff to achieve sufficient performance. 

This tradeoff invariably results in the occurrence of false positives which have a detrimental effect on the performance of the network functions using these data structures. 

Our project proposes to bring the notion of learned data structures (as explored in albeit in a different context) into switches to improve the false positive rate, given the same memory all the while still running at line-rate. This would open up the possibility for switches to scale to many more network functions and keep up with the demands of growing networks with their existing hardware. We plan to realize this proposal in the specific setting of software-defined networks.

