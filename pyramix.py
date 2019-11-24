import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
from itertools import combinations
from copy import deepcopy as dc
#import streamlit as st

move_position = {
        "U": [(0, 0), (3, 8), (2, 4)],
        "u": [(0, 0), (0, 1), (0, 2), (0, 3),
              (3, 8), (3, 3), (3, 7), (3, 6),
              (2, 4), (2, 6), (2, 5), (2, 1)],
        "L": [(2, 0), (1, 8), (0, 4)],
        "l": [(2, 0), (2, 1), (2, 2), (2, 3),
              (1, 8), (1, 3), (1, 7), (1, 6),
              (0, 4), (0, 6), (0, 5), (0, 1)],
        "R": [(3, 0), (0, 8), (1, 4)],
        "r": [(3, 0), (3, 1), (3, 2), (3, 3),
              (0, 8), (0, 3), (0, 7), (0, 6),
              (1, 4), (1, 6), (1, 5), (1, 1)],
        "B": [(1, 0), (2, 8), (3, 4)],
        "b": [(1, 0), (1, 1), (1, 2), (1, 3),
              (2, 8), (2, 3), (2, 7), (2, 6),
              (3, 4), (3, 6), (3, 5), (3, 1)],
    }

def pyraminx_puzzle(moves=[],puzz=None ):
    face_colours= ['R', 'G', 'Y', 'B']
    # move_position = { "U": [[0, 0], [3, 8], [2, 4]],
    #                   "u": [[0, 0], [0, 1], [0, 2], [0, 3], [3, 8], [3, 3], [3, 7], [3, 6], [2, 4], [2, 6], [2, 5], [2, 1]],
    #                   "L": [[2, 0], [1, 8], [0, 4]],
    #                   "l": [[2, 0], [2, 1], [2, 2], [2, 3], [1, 8], [1, 3], [1, 7], [1, 6], [0, 4], [0, 6], [0, 5], [0, 1]],
    #                   "R": [[3, 0], [0, 8], [1, 4]],
    #                   "r": [[3, 0], [3, 1], [3, 2], [3, 3], [0, 3], [0, 8], [0, 7], [0, 6], [1, 4], [1, 6], [1, 5], [1, 1]],
    #                   "B": [[1, 0], [2, 8], [3, 4]],
    #                   "b": [[1, 0], [1, 1], [1, 2], [1, 3], [2, 8], [2, 3], [2, 7], [2, 6], [3, 4], [3, 6], [3, 5], [3, 1]] }
    if puzz == None:
        puzzle = [[c for _ in range(9)] for c in face_colours]
    else:
        puzzle = dc(puzz)
    for move in reversed(moves):
        left = "'" not in move
        move_colors = { move: [puzzle[r][c] for r, c in move_position[move]] for move in move_position }
        current_color = shift_color(move_colors[move[0]], left)        
        for idx, pos in enumerate(move_position[move[0]]):
            puzzle[pos[0]][pos[1]] = current_color[idx]

    return puzzle

def shift_color(current_color, left):
    i = len(current_color) // 3
    if left:
        return current_color[i:] + current_color[0:i]
    return current_color[-i:] + current_color[:-i] 

def color(c):
    if c == 'R':return 'red'
    elif c =='G': return 'green'
    elif c =='Y': return 'yellow'
    elif c =='B': return 'blue'

def side(S):
    if S == set(['a','b','d']):return 0
    elif S == set(['a','b','c']):return 1
    elif S == set(['a','c','d']):return 2
    else: return 3


    
def show(puzzle):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    vertex = np.array([[0, 0, 0], [1, 0, 0], [1/2, np.sqrt(3/4), 0],  [1/2, np.sqrt(1/12), np.sqrt(3/4)]])
    vertexColour = {v:k for v,k in zip(['a','b','c','d'],vertex)}
    vEdge = {}
    vCenter = {}
    for v in list(combinations(vertexColour.keys(),2)):
        vEdge[v] = {}
        vEdge[v][v[1]]=(1/3)*vertexColour[v[0]]+(2/3)*vertexColour[v[1]]
        vEdge[v][v[0]]=(2/3)*vertexColour[v[0]]+(1/3)*vertexColour[v[1]]
    for v in list(combinations(vertexColour.keys(),3)):
        vCenter[v]=[(1/3)*vertexColour[v[0]]+(1/3)*vertexColour[v[1]]+(1/3)*vertexColour[v[2]]]
     
    ax.scatter3D(vertex[:, 0], vertex[:, 1], vertex[:, 2])
    face =[]
    colors=[]
    colorCount=[[0 for i in puzzle[0]] for j in puzzle]
    for v,l in vertexColour.items():
        for vs in list(combinations([(v1.keys(),v1[v]) for v0,v1 in vEdge.items() if v in v0],2)):
            newface = [l]+list([i[1] for i in vs])
            face.append(np.array(newface))
            pyraSide = side(set([j for i in list(vs) for j in list(i)[0]]))
            if pyraSide == 0:
                pyraPos = 0 if v=='d' else 4 if v=='a' else 8  
            elif pyraSide == 1:
                pyraPos = 0 if v=='c' else 4 if v=='b' else 8
            elif pyraSide == 2:
                pyraPos = 0 if v=='a' else 4 if v=='d' else 8
            else:
                pyraPos = 0 if v=='b' else 4 if v=='c' else 8
            colors.append(color(puzzle[pyraSide][pyraPos]))
            colorCount[pyraSide][pyraPos] += 1

    for vck,vcv in vCenter.items():
        for vs in [v1.items() for v0,v1 in vEdge.items() if v0[0] in vck and v0[1] in vck]:
            newface = dc(vcv)
            for vx in [i[1] for i in vs]:
                newface += [vx]
            face.append(np.array(newface))
            pyraSide = side(set(vck))
            pyraEdge = set([i[0] for i in vs])
            if pyraSide == 0:
                pyraPos = 1 if pyraEdge == set(['a','d']) else 3 if pyraEdge == set(['b','d']) else 6  
            elif pyraSide == 1:
                pyraPos = 1 if pyraEdge == set(['b','c']) else 3 if pyraEdge == set(['a','c']) else 6
            elif pyraSide == 2:
                pyraPos = 1 if pyraEdge == set(['a','d']) else 3 if pyraEdge == set(['a','c']) else 6
            else:
                pyraPos = 1 if pyraEdge == set(['b','c']) else 3 if pyraEdge == set(['b','d']) else 6
            colors.append(color(puzzle[pyraSide][pyraPos]))
            colorCount[pyraSide][pyraPos] += 1


    for vck,vcv in vCenter.items():
        for v in vertexColour.keys():
            if v not in vck: continue
            newface = dc(vcv)
            for vx in [v1[v0[0]] if v0[0] == v else v1[v0[1]] for v0,v1 in vEdge.items() if (v0[0] == v or v0[1] == v) and (v0[0] in vck and v0[1] in vck)]:
                newface += [vx]
            face.append(np.array(newface))
            pyraSide = side(set(vck))
            if pyraSide == 0:
                pyraPos = 2 if v == 'd' else 5 if v == 'a' else 7  
            elif pyraSide == 1:
                pyraPos = 2 if v == 'c' else 5 if v == 'b' else 7
            elif pyraSide == 2:
                pyraPos = 2 if v == 'a' else 5 if v == 'd' else 7
            else:
                pyraPos = 2 if v == 'b' else 5 if v == 'c' else 7
            colors.append(color(puzzle[pyraSide][pyraPos]))
            colorCount[pyraSide][pyraPos] += 1
    # print(colorCount)
    pc=Poly3DCollection(face, 
     facecolors=colors, linewidths=1, edgecolors='k', alpha=1)
    ax.add_collection3d(pc)

    return plt

if __name__ == '__main__':
    face_colors = ['R', 'G', 'Y', 'B']
    # opt = st.multiselect('What are your move',('u', 'b', 'l', 'r',"u'","b'","l'","r'"))
    # if opt == None:
    #     moves=[]
    # else:
    #     moves=[opt]
    puzzle = None
    for i in range(5):
        moves = ["u"] # Add moves here
        puzzle=pyraminx_puzzle(face_colors, moves,puzzle)
        show(puzzle).show()
