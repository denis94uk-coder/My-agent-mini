import local_ocr


captured = {}


class _Result:
    returncode = 0
    stdout = "Invoice 42\n"
    stderr = ""


def fake_run(command, **kwargs):
    captured["command"] = command
    captured.update(kwargs)
    return _Result()


original_which, original_run = local_ocr.shutil.which, local_ocr.subprocess.run
local_ocr.shutil.which = lambda command: "/usr/bin/tesseract"
local_ocr.subprocess.run = fake_run
try:
    assert local_ocr.extract_image(b"not-a-real-image") == "Invoice 42"
finally:
    local_ocr.shutil.which, local_ocr.subprocess.run = original_which, original_run
assert captured["command"][0] == "tesseract"
assert captured["command"][2:] == ["stdout", "-l", "eng"]
assert captured["timeout"] == 30
print("Local OCR helper checks passed")
