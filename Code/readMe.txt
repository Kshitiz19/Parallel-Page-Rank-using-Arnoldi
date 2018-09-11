1) pagerank.cu contains the main function, so all values to be printed can be handled using this file

2) You can decide the loop stopping criteria in pagerank.cu based on either iterations or based on difference between consecutive iteration values of Aq-q

3) gen.py is a script that generates a nxn sparse adjacency matrix that will represent a web graph matrix.
	run : python gen.py "value of n" > "outputfile-name"

4) final run : ./out "matrix-file"
