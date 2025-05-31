"""
This is a unit test module for the backend AI module.

It contains every type of AI available on backend of this application.
Each type of AI has its own module which should be possible to call individually but also be able to be called as a whole.
Each module of tests contains a file called testset.[txt|csv].
A testset.txt contains a list of test cases defined by its input query.
If it has testset.csv file also contains expected output which could either be a single value or range of minimum and maximum values, or a name of test and location of input file.
If there is an inputfile listed then the file should be loaded and used as input for the test as described within the modules __init__.py file.
"""
