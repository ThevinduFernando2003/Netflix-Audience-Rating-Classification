PYTHON := .venv/Scripts/python.exe

.PHONY: setup profile design-target tune train evaluate improve api ui test

setup:
	"C:\Users\User\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none\python.exe" -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

profile:
	$(PYTHON) scripts/profile_data.py

design-target:
	$(PYTHON) scripts/design_target.py

tune:
	$(PYTHON) scripts/tune.py

train:
	$(PYTHON) scripts/train.py

api:
	$(PYTHON) -m uvicorn src.serving.api:app --host 127.0.0.1 --port 8000

ui:
	$(PYTHON) -m streamlit run app/streamlit_app.py

evaluate:
	$(PYTHON) scripts/evaluate_test.py

improve:
	$(PYTHON) scripts/improve_cv.py

test:
	$(PYTHON) -m pytest
