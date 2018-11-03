from emdp.gridworld import txt_utilities
from emdp.examples.simple import _EXAMPLE_FOUR_ROOMS_TXT

LOCATION_OF_ROOM_FILE = './tests/example_room.txt'

# TODO(zaf): Make this into a TestCase class?

def test_get_char_matrix_from_strings():
    assert txt_utilities.get_char_matrix(_EXAMPLE_FOUR_ROOMS_TXT) is not None

def test_get_char_matrix_from_file():
    with open(LOCATION_OF_ROOM_FILE) as f_:
        assert txt_utilities.get_char_matrix(f_) is not None

def test_char_matrix_builder_integration():
    with open(LOCATION_OF_ROOM_FILE) as f_:
        char_matrix = txt_utilities.get_char_matrix(f_)
    assert txt_utilities.build_gridworld_from_char_matrix(char_matrix) is not None

def test_gridworld_correctly_built():
    with open(LOCATION_OF_ROOM_FILE) as f_:
        char_matrix = txt_utilities.get_char_matrix(f_)
    mdp, _ = txt_utilities.build_gridworld_from_char_matrix(char_matrix)

    # TODO(zaf): Add tests here to ensure that the world is what we expect it to be.
    # Do after zafarali/emdp/issues/8 is merged.



