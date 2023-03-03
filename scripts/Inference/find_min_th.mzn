include "globals.mzn";
include "alldifferent.mzn";

int: start;
int: end;
int: number_of_nodes;
array[1..number_of_nodes+2] of var 1..number_of_nodes+2: succ;


array[1..number_of_nodes+2, 1..number_of_nodes+2] of 0..100: edge_prediction;
array[1..number_of_nodes+2, 1..number_of_nodes+2] of var bool: edge_allowed;

constraint forall(i, j in 1..number_of_nodes+2 where i!=j) (edge_allowed[i,j] = (edge_prediction[i,j] <= warm_treshold[1]));
constraint forall(i in 1..number_of_nodes+2) (edge_allowed[i,i] = false);

constraint alldifferent(succ);
constraint succ[end] == start;
constraint circuit(succ);

constraint succ[start] != end;

constraint forall(i in 1..number_of_nodes+2) (succ[i] != i);

constraint forall(i in 1..number_of_nodes+1)(
forall(j in i+1..number_of_nodes+2)
 (
  if not (edge_allowed[i,j] \/ edge_allowed[j,i])
  then
    if i != end 
    then
      succ[i] != j 
    endif
    /\
    if j != end 
    then
      succ[j] != i 
    endif
  endif
));

constraint forall(i in 1..number_of_nodes+2 where i != end)(
  edge_allowed[i,succ[i]] \/ edge_allowed[succ[i], i]
);

constraint forall(i in 1..number_of_nodes+2 where i != end)(
  warm_treshold[1] >= min(j in 1..number_of_nodes+2) (edge_prediction[i,j])
);

array[1..1] of var 0..100: warm_treshold;
var 0..100: min_treshold;
constraint min_treshold = warm_treshold[1];

solve :: int_search(warm_treshold, input_order, indomain_min) minimize min_treshold;
