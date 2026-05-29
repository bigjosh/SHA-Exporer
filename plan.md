Our goal is to create a browser-based UI tool to help humans explore/understand the SHA256 algorithm.

# Representation

We will represent SHA256 as an acyclic directed circuit of two types of nodes: NAND gates and constants. Each node has a single bit value and that value can drive any number of input lower int he graph.

Every node has an ID and we will try to make the IDs be human readable. 

## Constant nodes

No inputs, the value is assigned - and can be modified at run time by the user.  

## NAND nodes
The current value is computed from two input nodes. 

## Output nodes

This represents and output of a circuit. It has a single input that points to another node, and the output node's value is a copy of the input. 

Output nodes can not be optimized away. The ensure that all the nodes that they depend on will remain in the graph.

# Construction of the graph

First we will create a graph of SHA256 first by directly translating the SHA256 operations into NAND circuits with algothim constants. The input bits are undefined at this step. The output of this step will be SHA256.nodes.

# Graph Optimization

Next we will make a graph compiler that aggressively optimizes a graph of nodes to minimize the total node count and also minimize the complexity (that is, loosely, the number of nodes each node has dependency on).

Since we will be using this graph for everything else, this step justifies a lot of effort to get it as optimal as possible. We should not stop until we are confident that we are within, say, 5% of the most optimal representation of the final graph.

Note that any node that is not an "output" node and is not used as in input to another node can always be deleted. 

Note that we may concatenate node files together before sending them into the compiler. So, for example, we might have a node file that sets the top n input bits to "0" and then prepend this to the SHA2567 node file, and then compile the resulting file to get an optimize circuit that has only 512-n free input bits and this will open up many more optimization opportunities. 

# Implementation details

We will only need to model a single SHA256 chunk, so the top of the graph will be the 512 input bits. These will have IDs in the form "IN-000", "IN-001", ... "IN-0511". 

The SHA256 algorithmic constants can appear scattered throughout the graph so they are next to the input they drive. The will IDs like "K-....".

The 256 output bits appear at the bottom of the graph and will have IDs like "OUT-000", ... "OUT-0255".

## Format

A txt file that contains one node per line- node ID, node type (N,C,O), and either (a 0 or 1 if the node is type C) or (a list of two comma sep input node IDs if type is N) or (a single input node ID if type O). 

SHA256.txt  file will  represent the SHA256 NAND and constant nodes for the SHA256. This is the input and output of the graph building tool. 

IN.txt file will be use to set the input bits using a list of constant nodes. 
