env_name=py310_challenge
conda_path=/opt/conda/bin/conda

url=$1
plat=$2
echo $url

mkdir challenge_test
cd challenge_test
wget $url

unzip challenge_code_${plat}.zip

$conda_path create -n $env_name python=3.10 -y

$conda_path run -n $env_name --no-capture-output pip install -r requirements.txt
$conda_path run -n $env_name --no-capture-output pip install deepquantum-v0.0.4/
$conda_path run -n $env_name --no-capture-output python scripts/runtest.py

$conda_path remove -n $env_name --all -y
