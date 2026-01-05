rm -rf websites/temp
mkdir websites/temp
/home/hgryoo/dev/ann-benchmarks/.venv/bin/python create_website.py --outputdir "websites/temp" --scatter --recompute

cd websites/temp
/home/hgryoo/dev/ann-benchmarks/.venv/bin/python -m http.server 8080
