import pytest
from dicom_edge_bench.selection import parse_selection, chunks


def test_all_first_range_and_duplicates():
    assert parse_selection("all", 4) == [0, 1, 2, 3]
    assert parse_selection("first:2", 4) == [0, 1]
    assert parse_selection("0:8:2", 8) == [0, 2, 4, 6]
    assert parse_selection("3,1,3", 4) == [3, 1, 3]
    assert parse_selection("uniform:3", 5) == [0, 2, 4]
    assert parse_selection("uniform:1", 5) == [0]
    assert list(chunks([0, 1, 2, 3, 4], 2)) == [[0, 1], [2, 3], [4]]


@pytest.mark.parametrize("spec", ["first:0", "first:6", "-1,2", "0:0", "0:8:0", "5"])
def test_invalid_selection(spec):
    with pytest.raises(ValueError):
        parse_selection(spec, 5)
