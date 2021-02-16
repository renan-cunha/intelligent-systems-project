import pytest
from flask import json
from api import app
import os

categorize_route = "v1/categorize"
valid_categories_set = {"Lembrancinhas", "Bebê", "Decoração", "Outros",
                        "Papel e Cia", "Bijuterias e Jóias"}

TEST_PRODUCTS_PATH = os.getenv("TEST_PRODUCTS_PATH")
with open(TEST_PRODUCTS_PATH, "r") as json_file:
    test_json = json.load(json_file)


def categorize_request(input_data):
    return app.test_client().post(categorize_route,
                                  data=json.dumps(input_data),
                                  content_type="application/json")


@pytest.mark.parametrize("input_data", [
        None,
        "",
        {},
        {"products": []},
        {"products": [{"title": ""}]},
        {"products": [{"concatenated_tags": ""}]},
        {"products": [{"other1": "", "other2": ""}]}
    ])
def test_request_with_invalid_data(input_data):
    response = categorize_request(input_data)

    assert response.status_code == 400
    assert response.data == b"(Bad Request)"


@pytest.mark.parametrize("input_data", [
    {"products": [{"title": None, "concatenated_tags": None}]},
    {"products": [{"title": "", "concatenated_tags": ""}]},
    {"products": [{"title": "", "concatenated_tags": "", "other": ""}]},
    {"products": [{"title": "a", "concatenated_tags": "a"},
                  {"title": "b", "concatenated_tags": "b"}]},
    test_json])
def test_request_with_valid_data(input_data):
    response = categorize_request(input_data)

    assert response.status_code == 200
    assert len(response.json["categories"]) == len(input_data['products'])
    assert set(response.json['categories']).issubset(valid_categories_set)
