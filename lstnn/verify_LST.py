'''
This code takes a 4 x 4 LST array and tries to solve it. It returns
the estimated relational complexity of the problem (Binary/Ternary/ Quaternary),
the proposed solution and what strategy was used to solve the problem.

- For Binary it just checks if three unique digits are in the targets row or column.
- For Ternary (assuming the above is not true) it checks if three unique digits are 
    in the targets row / column
- For Quaternary (assuming the above is not true) it employs a number of strategies 
    that I observed in the test dataset. I do think that this covers all possible 
    strategies in a 4 x 4 matrix but will quickly break down with larger matrices (or 
    become intractable)

'''

import numpy as np


def verify_LST(puzzle):
    '''
    This has been verified in the TEST dataset (i.e., the original LST items used
    in fMRI) but likely relies on several strategies that don't encompass
    all possible solutions for a puzzle (particularly in Three vector / Qua problems).
    It also won't scale up to other sized puzzles.

    return condition, solution, strategy
    '''
    row, col = np.where(puzzle == 5)

    # remove the target in the puzzle as this isn't information
    puzzle_nt = puzzle.copy()
    puzzle_nt[puzzle_nt == 5] = 0

    # One vector (binary)
    # binary within row
    solution = np.array((1, 2, 3, 4))
    condition = None
    strategy = None

    for q in solution:

        # check the target row
        if np.sum(puzzle_nt[row, :] == q):
            solution = np.delete(solution, np.where(solution == q)[0])
    if len(solution) == 1:
        condition = 'Binary'
        strategy = 'Binary-row'
        return condition, solution[0], strategy

    # binary within column
    solution = np.array((1, 2, 3, 4))
    for q in solution:

        # check the target column
        if np.sum(puzzle_nt[:, col] == q):
            solution = np.delete(solution, np.where(solution == q)[0])
    if len(solution) == 1:
        condition = 'Binary'
        strategy = 'Binary-col'
        return condition, solution[0], strategy

    # Two vectors (Ternary)
    solution = np.array((1, 2, 3, 4))
    for q in solution:

        # check the target row
        if np.sum(puzzle_nt[row, :] == q):
            solution = np.delete(solution, np.where(solution == q)[0])

        # check the target column
        if np.sum(puzzle_nt[:, col] == q):
            solution = np.delete(solution, np.where(solution == q)[0])

    if len(solution) == 1:
        condition = 'Ternary'
        strategy = 'Ternary'
        return condition, solution[0], strategy

    # Three vectors (Quaternary)
    # by row
    solution = np.array((1, 2, 3, 4))
    for q in solution:
        # check the target row
        if np.sum(puzzle_nt[row, :] == q):
            solution = np.delete(solution, np.where(solution == q)[0])

    # NOT steps in empty cells within row
    # find blank cells and sum within that column
    idx = (puzzle[row, :] == 0)[0]
    for q in solution:

        # if no possible solutions have been eliminated
        # try looking for common solution across three
        # remaining blanks
        if len(solution) == 4 and np.sum(puzzle[:, idx] == q) == 3:
            solution = q
            condition = 'Quaternary'
            strategy = 'Quaternary-3col'
            return condition, solution, strategy

        # if one possible solution has been eliminated
        # try looking for common solution across two blanks
        if len(solution) == 3 and np.sum(puzzle[:, idx] == q) == 2:
            solution = q
            condition = 'Quaternary'
            strategy = 'Quaternary-row-2col'
            return condition, solution, strategy

    # by column
    solution = np.array((1, 2, 3, 4))
    for q in solution:
        # check the target col
        if np.sum(puzzle_nt[:, col] == q):
            solution = np.delete(solution, np.where(solution == q)[0])

    # NOT steps in empty cells within row
    # find blank cells and sum within that column
    idx = (puzzle[:, col] == 0).T[0]
    for q in solution:

        #(see above for logic)
        if len(solution) == 4 and np.sum(puzzle[idx, :] == q) == 3:
            solution = q
            condition = 'Quaternary'
            strategy = 'Quaternary-3row'
            return condition, solution, strategy

        if len(solution) == 3 and np.sum(puzzle[idx, :] == q) == 2:
            solution = q
            condition = 'Quaternary'
            strategy = 'Quaternary-col-2row'
            return condition, solution, strategy
    return condition, solution, strategy
