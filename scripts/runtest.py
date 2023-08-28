import os
import subprocess

root = os.path.dirname(os.path.dirname(__file__))

def run_cmd(cmd, check=True, cwd=None):
    subprocess.run(cmd.split(), check=check, cwd=cwd)


run_cmd('python main.py dj', cwd=os.path.join(root, 'p1'))
run_cmd('python main.py pqc', cwd=os.path.join(root, 'p1'))
run_cmd('python main.py', cwd=os.path.join(root, 'p2'))
print('* 测试完成！')
