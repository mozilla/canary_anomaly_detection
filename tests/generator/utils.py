def assert_two_data_dicts_equal(dict_true, dict_pred):
    assert isinstance(dict_pred, dict)
    assert dict_pred.keys() == dict_true.keys()
    for key in dict_pred.keys():
        if isinstance(dict_pred[key], dict):
            assert_two_data_dicts_equal(dict_true[key], dict_pred[key])
        else:
            try:
                assert dict_pred[key] == dict_true[key]
            except ValueError:
                assert all(dict_pred[key] == dict_true[key])
