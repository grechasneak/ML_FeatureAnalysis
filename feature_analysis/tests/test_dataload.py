
'''
This module will test.

'''


# content of test_sample.py
def inc(x):
    if x < 0:
        return x - 1
    return x + 1

def test_answer():
    assert inc(3) == 4

def test_answer_negative():
    assert inc(-2) == -3




