if [ ! -d "./venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
python3 -m pip install --upgrade build
python3 -m build

python3 -m pip install --upgrade twine
python3 -m twine upload dist/*
