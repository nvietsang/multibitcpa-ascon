import pycryptosat as cs
import sys
# Cardinality constraint system proposed by Sinz # sSat: the CNF system used
# setId: list of the variable id of the set
# nb_sets: number of sets
# cc: cardinality constraint
# startextra: index of additional variable unused so far, can be a value large enough
# For more details C. Sinz, Towards an Optimal CNF Encoding of Boolean Cardinality Constraints.
def Cardinality_Constraints(sSat,setId,nb_sets,cc,startextra):
    sSat.add_clause([-(setId[0]), startextra]) 
    for j in range (2,cc+1):
        sSat.add_clause([-(startextra+j-1)]) 
    for i in range(2,nb_sets):
        sSat.add_clause([-(setId[i-1]), startextra+(cc)*(i-1)])
        sSat.add_clause([-(startextra+(cc)*(i-2)), startextra+(cc)*(i-1)])
        for j in range(2,cc+1): 
            sSat.add_clause([-(setId[i-1]),
                            -(startextra+cc*(i-2)+j-2),
                            startextra+(i-1)*cc+j-1]) 
            sSat.add_clause([-(startextra+cc*(i-2)+j-1),
                            startextra+(i-1)*cc+j-1]) 
            
        sSat.add_clause([-(setId[i-1]),
                        -(startextra+cc*(i-2)+cc-1)]) 
    sSat.add_clause([-(setId[nb_sets -1]),
                     -(startextra+cc*(nb_sets -2)+cc-1)]) 
    startextra+=(nb_sets -1)*cc


if __name__ == "__main__": 
    # Arguments
    # Run ‘python3 01_nbcparuns.py 64 36 45 14 2‘ for k0z0
    # Run ‘python3 01_nbcparuns.py 64 57 23 14 2‘ for k0z4
    # Run ‘python3 01_nbcparuns.py 64  3 25 14 2‘ for k1z1
    n  = int(sys.argv[1]) # n = 64 indexes
    s1 = int(sys.argv[2]) # 1st shift
    s2 = int(sys.argv[3]) # 2nd shift
    ub = int(sys.argv[4]) # upper bound for number of subsets
    nd = int(sys.argv[5]) # bitlen
    # Compute all subset
    list_sets =[]
    restriction = [31-j for j in range(nd-1)] + [63-j for j in range(nd-1)]
    print(f"Restriction: {restriction}")
    for i in range(n):
        if i in restriction: continue
        _set = []
        for j in range(nd):
            _set.append((i+j)%n)
            _set.append((i+s1+j)%n)
            _set.append((i+s2+j)%n)
        list_sets.append(_set)

    # Save the universe
    universe=list(range(n))
    # Create the solver object
    sSat=cs.Solver()
    # Add constraint
    # For each element of the universe, we should select at least one subset containing the element
    for i in universe:
        # create an empty disjuction 
        clause_presence =[]
        for _set in list_sets:
            # if subset j contains the element i, we add the variable corresponding to the set j to the disjunctions
            if i in _set:
                clause_presence += [list_sets.index(_set)+1]
        # One variable in clause_presence must be set to true 
        # Add disjunction to the conjunction 
        sSat.add_clause(clause_presence)

    # Cardinality constraints
    Cardinality_Constraints(sSat, 
                            list(range(1,len(list_sets)+1)),
                            len(list_sets), 
                            ub, 
                            len(list_sets)+1)
    # Solve it
    satq,solution=sSat.solve()
    # SAT or UNSAT
    print(satq)
    # if SAT print the solution 
    if(satq):
        # for all variable corresponding to a subset
        for i in range(1,len(list_sets)+1):
            # if True then the subset has been selected
            if solution[i]: print(list_sets[i-1])
            # else the subset has not been selected each element of this subset should appear in at least one other selected subset

# Results

# For nd = 1
# indexes_k0b1z0 = [1, 2, 5, 8, 11, 12, 15, 18, 19, 22, 25, 28, 29, 32, 35, 39, 42, 45, 49, 52, 55, 59, 62]     # 23
# indexes_k0b1z4 = [1, 6, 7, 9, 11, 12, 17, 22, 23, 25, 27, 28, 33, 34, 38, 39, 43, 44, 48, 49, 54, 55, 59, 60] # 24
# indexes_k1b1z1 = [3, 5, 6, 7, 13, 15, 17, 22, 24, 26, 32, 33, 34, 39, 41, 43, 45, 50, 51, 52, 53, 58, 60, 62] # 24

# For nd = 2
# indexes_k0b2z0 = [2, 10, 19, 22, 24, 26, 36, 40, 42, 44, 49, 51, 53, 56] # 14
# indexes_k0b2z4 = [3, 9, 14, 19, 24, 30, 35, 41, 46, 51, 56, 62]          # 12
# indexes_k1b2z1 = [2, 10, 11, 20, 22, 28, 30, 37, 39, 43, 46, 48, 55, 57] # 14

# For nd = 3
# indexes_k0b3z0 = [22, 28, 34, 40, 45, 48, 41, 43, 56, 61] # 10
# indexes_k0b3z4 = [20, 23, 33, 36, 39, 42, 45, 48, 51, 60] # 10
# indexes_k1b3z1 = [1, 7, 13, 20, 29, 32, 39, 48, 58]       # 9

# For nd = 4
# indexes_k0b4z0 = [28, 32, 33, 45, 49, 53, 57, 60] # 8
# indexes_k0b4z4 = [10, 21, 32, 36, 40, 48, 58]     # 7
# indexes_k1b4z1 = [1, 8, 18, 22, 26, 36, 54]       # 7