import pytest
import anu_ctlab_io.netcdf.dict_transformer as dt
from copy import deepcopy


def setup_dictionary_transformer() -> (dict, dt.DictTransformer):
    d = {'a': 1}
    dtx = dt.DictTransformer(d)
    return (d, dtx)


def test_dictionary_transformer_insert():
    d, dtx = setup_dictionary_transformer()
    # inserting at a pre-existing key must be invalid for insert to be a lossless transform
    with pytest.raises(dt.DictTransformerException):
        dtx.insert({'a': 1})
    dtx.insert({'b' : {'c' : 2}})
    dtx.insert({'b' : {'d' : 3}})
    # must be able to insert multiple keys into a nested dict
    assert d == {'a' : 1, 'b' : {'c' : 2, 'd' : 3}}


def test_dictionary_transformer_remove():
    d, dtx = setup_dictionary_transformer()
    # removing nested keys must work
    dtx.insert({'a': {'b': 1}})
    dtx.remove([['a', 'b']])
    assert d == {}
    dtx.undo()
    assert d == {'a': {'b': 1}}
    dtx.redo()
    assert d == {}


def test_dictionary_transformer_update():
    d, dtx = setup_dictionary_transformer()
    dtx.insert({'a': {'b': 1, 'c': 2}})
    dtx.update({'a': {'b': 3}})
    assert d == {'a': {'b': 3, 'c': 2}}
    dtx.undo()
    assert d == {'a': {'b': 1, 'c': 2}}


def test_dictionary_transformer_rekey():
    d, dtx = setup_dictionary_transformer()
    dtx.insert({'b': 2})
    dtx.insert({'c': 3})
    dtx.rekey({'a': 'b', 'b': 'c', 'c': 'a'})
    assert d == {'a' : 3, 'b' : 1, 'c' : 2}, "Rekey needs to allow for overlap in renamed keys!"
    d, dtx = setup_dictionary_transformer()
    dtx.rekey({'a': ['a', 'a']})
    assert d == {'a' : {'a': 1}}, "Rekey must be able to create nested dicts!"


def test_dictionary_transformer_undo_redo():
    d, dtx = setup_dictionary_transformer()
    # apply every operation and show undo and redo work correctly for them
    dtx.insert({'b' : 2, 'c' : 3})
    dtx.remove(['a'])
    dtx.rekey({'b' : 'bb', 'c' : 'cc'})
    dtx.update({'cc': 4})
    assert d == {'bb' : 2, 'cc' : 4}
    dtx.undo()
    assert d == {'bb' : 2, 'cc' : 3}
    dtx.redo()
    assert d == {'bb' : 2, 'cc' : 4}
    dtx.undo_all()
    assert d == {'a': 1}
    dtx.redo_all()
    assert d == {'bb' : 2, 'cc' : 4}


def test_dictionary_transformer_overwritting():
    d, dtx = setup_dictionary_transformer()
    dtx.insert({'b': 2})
    dtx.insert({'c': 3})
    assert d == {'a' : 1, 'b' : 2, 'c' : 3}
    dtx.undo_all()
    dtx.insert({'d': 4})
    dtx.redo_all()
    assert d == {'a': 1, 'd': 4}, "Making edits with history rolled back should clear redo-ability."


def test_dictionary_transformer_from_existing():
    d, dtx = setup_dictionary_transformer()
    dtx.insert({'b': 2})
    d_cpy = deepcopy(d)
    dtx_cpy = dt.DictTransformer.from_existing_transformer(d_cpy, dtx)
    assert d_cpy == {'a': 1, 'b': 2}
    dtx_cpy.undo()
    assert d_cpy == {'a': 1}
