venv:
	python -m venv $@

requirements: venv
	. venv/bin/activate; pip install -r requirements.txt

notebook = 'eugene-address-wealth.ipynb'

html:
	. venv/bin/activate; jupyter nbconvert --to html --no-input ${notebook}

slides:
	. venv/bin/activate; jupyter nbconvert --to slide --no-input ${notebook}
