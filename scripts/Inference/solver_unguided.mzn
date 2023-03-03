include "globals.mzn";
include "alldifferent.mzn";

int: start;
int: end;
int: number_of_nodes;
array[1..number_of_nodes+2] of var 1..number_of_nodes+2: succ;


array[1..number_of_nodes+2, 1..number_of_nodes+2] of float: matrix;


constraint alldifferent(succ);
constraint succ[end] == start;
constraint circuit(succ);

var float: sum_path;
constraint sum_path = sum(i in 1..number_of_nodes+2 where i != end) (matrix[i,succ[i]]);
  
solve minimize sum_path;

