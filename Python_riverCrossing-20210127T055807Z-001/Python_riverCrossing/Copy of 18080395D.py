def genStates():
    '''
    input: two boolean values E and W
    output: all possible states
    '''
    direction=("E","W")                     #E-the river bank, W- the hotel
    states=[]
    for i in direction:
        for j in direction:
            for k in direction:
                for l in direction:
                    for m in direction:
                        for n in direction:
                            for o in direction:
                                astate=i+j+k+l+m+n+o
                                states.append(astate)   #the possible state of each person and boat stored.
    return states

def isAStateLegal(state):
    '''
    input: a state out of all posssible states.
    output: whether the state is legal or not.
    '''
    legal=[]
    illegal=[]
    i=state
    if (i[0]==i[3] or i[0]==i[5]) and i[0]!=i[1]:   #Green wife cannot be with other men without her husband
        return False
    elif (i[2]==i[1] or i[2]==i[5]) and i[2]!=i[3]: #Blue wife cannot be with other men without her husband
        return False
    elif (i[4]==i[1] or i[4]==i[3]) and i[4]!=i[5]: #Red wife cannot be with other men without her husband
        return False
    else:
        return True

def nextStates(current,legal):
    '''
    input: a current legal state and rest all other legal states.
    output: the legal states to which the current legal state can go.
    '''
    neighbors=set()
    index=set()
    temp=set()
    temp1=set()
    for i in legal:
        if i[6]!=current[6]:    #state of boat of current legal state and the other legal state should not be the same, else how will the state of people change.
            if current.count('W')==1 and current[6]!='W':   #case1: if there is only one person(west) on one side and boat on the other side(east).
                for j in range(0,6):
                    if current[j]=='W':
                        index=j
                if current[index]==i[index]:
                    if current[:6].count('E')-i[:6].count('E')==1 or current[:6].count('E')-i[:6].count('E')==2: #only two people can change the state because only two people can sit on the boat.
                        neighbors.add(i)
            elif current.count('E')==1 and current[6]!='E': #case2: if there is only one person(east) on one side and boat on the other side(west).
                for j in range(0,6):
                    if current[j]=='E':
                        index=j
                if current[index]==i[index]:
                    if current[:6].count('W')-i[:6].count('W')==1 or current[:6].count('W')-i[:6].count('E')==2: #only two people can change the state because only two people can sit on the boat.
                        neighbors.add(i)
                        
            elif current[:6].count('W')>1 and current[6]=='W':  #case3: if there are more than two people on one side(west) with the boat.
                
                for x in range(0,6):
                    if current[x]=='E':
                        index.add(x)
                diff=0
                
                for y in range(0,6):
                    if current[y]!=i[y]:
                        diff=diff+1
                if diff==1 or diff==2:  #only two people can change the state because only two people can sit on the boat.
                    temp.add(i)         #temporarily the possible next legal states are added.
            elif current[:6].count('E')>1 and current[6]=='E':  #case4: if there are more than two people on one side(east) with the boat.
                
                for x in range(0,6):
                    if current[x]=='W':
                        index.add(x)
                diff=0
                
                for y in range(0,6):
                    if current[y]!=i[y]:
                        diff=diff+1
                if diff==1 or diff==2:  #only two people can change the state because only two people can sit on the boat.
                    temp.add(i)        #temporarily the possible next legal states are added.  
        else:
            if current[:6].count('E')==1 and i[6]=='E': #case5: if there is only one person on one side(east) with the boat.
                neighbors.add('WWWWWWW')
                break
            elif current[:6].count('W')==1 and i[6]=='W': #case6: if there is only one person on one side(west) with the boat.
                neighbors.add('EEEEEEE')
                break
    for check in temp:      # the people who are on the same side of the boat can change their states. 
        for h in index:
            if current[h]!=check[h]:
                temp1.add(check)    #purpose: to remove those cases where the people have changed their states when boat was not on their side. These unwanted cases are stored temporarily.
    temp2=set()
    temp2=temp-temp1    #purpose: to remove those cases where the people have changed their states when boat was not on their side, fulfilled.
    for f in temp2:
        neighbors.add(f)
    
    if current=='WWWWWWW':
        neighbors=neighbors-neighbors   #Once everyone has reached the hotel, they will not go to the other side.
        
    return neighbors

def genGraph(S):
    '''
    input: all possible states
    output: a graph showing- to which states each particular state can go.
    '''
    
    setLegalStates = []
    graph={}
    
    for n in range(len(S)):
        if isAStateLegal(S[n]) == True:     #if a state is legal, it will be appended in the list of legal states.
            setLegalStates.append(S[n])     
          
    for n in range(len(setLegalStates)):
        setNextNodes = nextStates(setLegalStates[n],setLegalStates) #each state is being created as a node and the respective possible neighbor states are assigned to them.
        graph.update({setLegalStates[n]:setNextNodes})
        
    
    return graph

def findShortestPath(graph, start, end, path=[]):
    '''
    input: a start node, an end node and a graph.
    output: the shortest path to rech the end node from the start node.
    '''
    path = path + [start]
    if start == end:
        return path

    if not (start in graph):
        return None
    shortestPath = None
    for node in graph[start]:
        if node not in path:
            newpath = findShortestPath(graph, node, end, path)
            if newpath:
                if not shortestPath or len(newpath) < len(shortestPath):
                    shortestPath = newpath
    return shortestPath

def printPath(path):
    '''
    input: the shortest path.
    output: presenting the shortest path to the user in a readable manner.
    '''
    i=0
    for i in range(0,len(path)-1):
        people=[]
        direction=0
        for j in range(0,6):
            if path[i][j]!=path[i+1][j]:
                people.append(j)                        #if the person is changing state, his/her index number is being stored.
                if path[i][j]=='E':                     #if person is in the east, he/she has to go to the west.
                    direction='went from east to west'
                elif path[i][j]=='W':                   #if person is in the east, he/she has to go to the west.
                    direction='went from west to east'

        #index numbers representing different people:
        #for the first person changing the state:
                    
        if people[0]==0:
            person1='Green wife'
        elif people[0]==1:
            person1='Green husband'
        elif people[0]==2:
            person1='Blue wife'
        elif people[0]==3:
            person1='Blue husband'
        elif people[0]==4:
            person1='Red wife'
        elif people[0]==5:
            person1='Red husband'

    
         #if two people are changing their states, for the second person:
            
        if len(people)==2:

            #index numbers representing different people:
            
            if people[1]==0:
                person2='Green wife'
            elif people[1]==1:
                person2='Green husband'
            elif people[1]==2:
                person2='Blue wife'
            elif people[1]==3:
                person2='Blue husband'
            elif people[1]==4:
                person2='Red wife'
            elif people[1]==5:
                person2='Red husband'
            i=i+1
            print(i,person1,'and',person2,direction,'.')
        
        elif len(people)==1:
            i=i+1
            print(i,person1,direction,'.')
        person1=0
        person2=0
        direction=0
        people=[]
    

                
def solver():
    '''
    input: no input
    output: to show the shortest path to the reader in a readable manner.
    '''
    setAllStates = genStates()          
    G = genGraph(setAllStates)
    src = "EEEEEEE"                        
    des = "WWWWWWW"                        

    path = findShortestPath(G,src,des)  
                                       
    printPath(path)
    return
solver()
