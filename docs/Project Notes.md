Trying to follow https://goodresearch.dev/

# Using / Installing src
pip install -e .


# TO DO
- Consider a more stringent training set. Perhaps where the critical information in the puzzle is not repeated in training. This would limit binary a lot where there is only 4^3 possible organisations and 36 of them are already taken...
- There is a weird interaction between dist (similarity between training and testing) the number of puzzles (which is currently related to dist, I guess) and the batch size used for training.
    - Seems like it is largely due to the number of puzzles, rather than the dist which is encouraging