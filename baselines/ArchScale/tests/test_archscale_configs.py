from baselines.ArchScale.models.factory import build_config, describe_layer_schedule


def test_sambay_d8_config_matches_paper_shape():
    config = build_config("sambay")
    assert config.name == "sambay_d8"
    assert config.n_layer == 8
    assert config.n_embd == 992
    assert config.n_head == 8
    assert config.n_query_groups == 2
    assert config.head_size == 128
    assert config.intermediate_size == 3968
    assert config.yoco is True
    assert config.gmu_yoco is True


def test_sambayoco_d8_config_matches_paper_shape():
    config = build_config("sambayoco")
    assert config.name == "sambayoco_d8"
    assert config.n_layer == 8
    assert config.n_embd == 1008
    assert config.n_head == 8
    assert config.n_query_groups == 2
    assert config.head_size == 128
    assert config.intermediate_size == 4032
    assert config.yoco is True
    assert config.gmu_yoco is False


def test_d8_layer_schedules_expose_expected_difference():
    sambay = describe_layer_schedule("sambay")
    sambayoco = describe_layer_schedule("sambayoco")

    assert [item["mixer"] for item in sambay] == [
        "mamba",
        "sliding_window_attention",
        "mamba",
        "sliding_window_attention",
        "mamba",
        "full_attention",
        "gmu",
        "cross_attention",
    ]
    assert [item["mixer"] for item in sambayoco] == [
        "mamba",
        "sliding_window_attention",
        "mamba",
        "sliding_window_attention",
        "mamba",
        "full_attention",
        "cross_attention",
        "cross_attention",
    ]
