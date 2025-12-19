"""
All credit for Sudoku related code goes to:
- https://github.com/Kyubyong/sudoku/blob/master/generate_sudoku.py
    - (Kyubyong Park. kbpark.linguist@gmail.com www.github.com/kyubyong)
    - (Which in turn was adapted from https://www.ocf.berkeley.edu/~arel/sudoku/main.html.)
I have adapted this slightly to suit a 4 x 4 LST rather than a fully fledged sudoku. All credit
goes to the above repos.
"""
import random
import copy
import numpy as np


sample = [[1, 2, 3, 4],
          [4, 1, 2, 3],
          [3, 4, 1, 2],
          [2, 3, 4, 1]]

"""
Randomly arrange numbers in a grid while making all rows, columns and
squares (sub-grids) contain the numbers 1 through 9.
For example, "sample" (above) could be the output of this function. """


def construct_puzzle_solution():
    # Loop until we're able to fill all 81 cells with numbers, while
    # satisfying the constraints above.
    while True:
        try:
            puzzle = [[0]*4 for i in range(4)]  # start with blank puzzle
            rows = [set(range(1, 5)) for i in range(4)]  # set of available
            columns = [set(range(1, 5)) for i in range(4)]  # numbers for each
            #squares = [set(range(1,5)) for i in range(4)] #   row, column and square
            for i in range(4):
                for j in range(4):
                    # pick a number for cell (i,j) from the set of remaining available numbers
                    #choices = rows[i].intersection(columns[j]).intersection(squares[int(np.floor((i/3)*3 + j/3))])
                    choices = rows[i].intersection(columns[j])

                    choice = random.choice(list(choices))

                    puzzle[i][j] = choice

                    rows[i].discard(choice)
                    columns[j].discard(choice)
                    #squares[int(np.floor((i/3)*3 + j/3))].discard(choice)

            # success! every cell is filled.
            return puzzle

        except IndexError:
            # if there is an IndexError, we have worked ourselves in a corner (we just start over)
            pass


"""
Randomly pluck out cells (numbers) from the solved puzzle grid, ensuring that any
plucked number can still be deduced from the remaining cells.
For deduction to be possible, each other cell in the plucked number's row, column,
or square must not be able to contain that number. """


def pluck(puzzle, n=0):
    """
    Answers the question: can the cell (i,j) in the puzzle "puz" contain the number
    in cell "c"? """
    def canBeA(puz, i, j, c):
        v = puz[int(np.floor(c/4))][int(np.floor(c % 4))]
        if puz[i][j] == v:
            return True
        if puz[i][j] in range(1, 5):
            return False

        for m in range(4):  # test row, col, square
            # if not the cell itself, and the mth cell of the group contains the value v, then "no"
            if not (m == c/4 and j == c % 4) and puz[m][j] == v:
                return False
            if not (i == c/4 and m == c % 4) and puz[i][m] == v:
                return False

        return True

    """
    starts with a set of all 81 cells, and tries to remove one (randomly) at a time
    but not before checking that the cell can still be deduced from the remaining cells. """
    cells = set(range(16))
    cellsleft = cells.copy()
    while len(cells) > n and len(cellsleft):
        # choose a cell from ones we haven't tried
        cell = random.choice(list(cellsleft))
        cellsleft.discard(cell)  # record that we are trying this cell

        # row, col and square record whether another cell in those groups could also take
        # on the value we are trying to pluck. (If another cell can, then we can't use the
        # group to deduce this value.) If all three groups are True, then we cannot pluck
        # this cell and must try another one.
        #row = col = square = False
        row = col = False
        for i in range(4):
            if i != int(np.floor(cell/4)):
                if canBeA(puzzle, i, int(np.floor(cell % 4)), cell):
                    row = True
            if i != int(np.floor(cell % 4)):
                if canBeA(puzzle, int(np.floor(cell/4)), i, cell):
                    col = True
            #if not (((cell/9)/3)*3 + i/3 == cell/9 and ((cell/9)%3)*3 + i%3 == cell%9):
            #    if canBeA(puzzle, ((cell/9)/3)*3 + i/3, ((cell/9)%3)*3 + i%3, cell): square = True

        if row and col:
            continue  # could not pluck this cell, try again.
        else:
            # this is a pluckable cell!
            puzzle[int(np.floor(cell/4))][int(np.floor(cell % 4))
                                          ] = 0  # 0 denotes a blank cell
            # remove from the set of visible cells (pluck it)
            cells.discard(cell)
            # we don't need to reset "cellsleft" because if a cell was not pluckable
            # earlier, then it will still not be pluckable now (with less information
            # on the board).

    # This is the puzzle we found, in all its glory.
    return (puzzle, len(cells))


"""
That's it.
If we want to make a puzzle we can do this:
    pluck(construct_puzzle_solution())
    
The following functions are convenience functions for doing just that...
"""


"""
This uses the above functions to create a new puzzle. It attempts to
create one with 28 (by default) given cells, but if it can't, it returns
one with as few givens as it is able to find.
This function actually tries making 100 puzzles (by default) and returns
all of them. The "best" function that follows this one selects the best
one of those.
"""


def run(n=5, iter=100):
    all_results = {}
#     print "Constructing a sudoku puzzle."
#     print "* creating the solution..."
    a_puzzle_solution = construct_puzzle_solution()

#     print "* constructing a puzzle..."
    for i in range(iter):
        puzzle = copy.deepcopy(a_puzzle_solution)
        (result, number_of_cells) = pluck(puzzle, n)
        all_results.setdefault(number_of_cells, []).append(result)
        if number_of_cells <= n:
            break

    return all_results, a_puzzle_solution


def best(set_of_puzzles):
    # Could run some evaluation function here. For now just pick
    # the one with the fewest "givens".
    return set_of_puzzles[min(set_of_puzzles.keys())][0]


def display(puzzle):
    for row in puzzle:
        print(' '.join([str(n or '_') for n in row]))
