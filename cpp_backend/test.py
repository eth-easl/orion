import os
import subprocess

for i in range(20):
    try:
        print(i)
        subprocess.check_output(['python', 'launch_jobs.py', 'examples/basic_config1.json'])
    except subprocess.CalledProcessError as e:
        s = e.output.decode('utf-8')
        print(type(s))
        lines = s.split("\n")
        for line in lines:
            print(line)
        print("FAILED!")
        exit(1)

print("OK!")

#for i in range(10):
#    os.system("python launch_jobs.py examples/basic_config1.json")
