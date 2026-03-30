import numpy as np

#Step 1: Initialize variables (for min problems only)
#for max problems multiply c by -1 and the final z by -1 too
c = np.array([-3, -1, 0, 0])

A = np.array([
    [1, 1, 1, 0],
    [2, 1, 0, 1]
])

b = np.array([2, 4])

#In the B list put as many variables as restrictions you have
#Make sure that the array these 2 variables make can be reversed
#In this example take the 2 and 3rd columns of the A array which is 
#[[1, 0], [0, 1]] which is reversible. Note that python counts from 0 so the 
#2nd column is the actual third one.
B = [2, 3]
N = [0, 1]

while True:
    #Create the necessary arrays
    #cB = c[B] works by grabbing data from c only in the indexes that are in B
    #in this example cB becomes [0, 0] 
    cB = c[B]
    cN = c[N]
    #Same logic applies here
    #The : grabs all elements from the columns that B has in it
    #In this example A[:, 2] grabs the 3rd column from A so [1, 0]^T
    AB = A[:, B]
    AN = A[:, N]

    #Step 2: Calculate reverse array and the basic solution
    AB_reverse = np.linalg.inv(AB)
    xB = AB_reverse @ b

    #Step 3: Calculate the dual and slack variables
    w_transpose = cB.transpose() @ AB_reverse
    sN = cN.transpose() - (w_transpose @ AN)

    #Step 4: Check solution
    #If sN has no negative elements then xB is the best solution and z the min value
    if(np.min(sN) >= 0):
        print(f"Variables: ${xB}")
        z = cB @ xB
        print(f"Min value is: ${z}")
        break;
    #If sN has at least one negative element then the solution can get better
    else: 
        #Dantzig rule: Incoming variable
        #Find the position of the min element of sN and put it in t
        t = np.argmin(sN)
        #l is the incoming variable which will be inserted into B in next steps
        l = N[t]

        #Outgoing variable
        #hl is the pivot column
        hl = AB_reverse @ A[:, l] 
        #If all hl elements are below 0 then the problem is unbounded and has no best solution
        if(np.max(hl) <= 0):
            print("Unbounded problem")
            break;
        #If all hl elements are above 0 then we find the outgoing variable
        else: 
            #Here we make a new array ratios with the same shape as xB and every element as infinite
            ratios = np.full(xB.shape, np.inf)
            
            #This mask acts as a boolean restriction to ensure hl is always above 0 so that it doesnt divide by 0
            mask = hl > 0
            #Get all the division results into ratios
            ratios[mask] = xB[mask] / hl[mask]
            #Find the position of the min element of the ratios array
            r = np.argmin(ratios)
            #K is the outgoing variable
            k = B[r]

            #Put the incoming variable into B and the outgoing into N
            B[r] = l
            N[t] = k

            #After this the loop repeats 

