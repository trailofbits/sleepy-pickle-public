from ..demos.simple_nn_backdoor import add_indentation, xor_obfuscate

# Tests for add_indentation
def test_add_indentation_empty():
    assert add_indentation([]) == ''

def test_add_indentation_multiple_lines():
    lines = ["line1", "line2", "line3"]
    expected = "    line1\n    line2\n    line3\n"
    assert add_indentation(lines) == expected

def test_add_indentation_special_chars():
    lines = ["", " ", "line with space", "line_with_underscore"]
    expected = "    \n     \n    line with space\n    line_with_underscore\n"
    assert add_indentation(lines) == expected

# Tests for xor_obfuscate
def test_xor_obfuscate_basic():
    input_string = "test"
    key = 123
    obfuscated = xor_obfuscate(input_string, key)
    expected_obfuscated = xor_obfuscate(input_string.encode(), key)
    assert obfuscated == expected_obfuscated

def test_xor_obfuscate_empty_string():
    assert xor_obfuscate("", 123) == "b''"

def test_xor_obfuscate_unicode():
    input_string = "测试"
    key = 55
    obfuscated = xor_obfuscate(input_string, key)
    expected_obfuscated = xor_obfuscate(input_string.encode(), key)
    assert obfuscated == expected_obfuscated

def test_xor_obfuscate_deobfuscate():
    input_string = "another test string"
    key = 45
    obfuscated = xor_obfuscate(input_string, key)
    expected_obfuscated = xor_obfuscate(input_string.encode(), key)
    assert obfuscated == expected_obfuscated