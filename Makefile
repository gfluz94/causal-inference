install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt &&\
	python setup.py install &&\
	cd ..

format:
	python3 -m black .

lint:
	python3 -m pylint --disable=R,C,W0201 causal_inference
test:
	python3 -m pytest -vv --cov