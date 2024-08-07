# PBR

## Update
```
# make changes & commit
python setup.py sdist
# note generated version X.devY and "bump"
git tag -am "Version 0.0.6" 0.0.6
```

## Test
```
cd /tmp
virtualenv venv
. venv/bin/activate
pip install $HOME/dist/pytimewarp-0.0.6.tar.gz
```

## Push After Testing
```
ggpush --follow-tags
```
