import os
import subprocess
import numpy as np

# execute command, and return the output
def execCmd(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text

# write "data" to file-filename
def writeFile(filename, data):
    f = open(filename, "w")
    for i in data:
        for j in i:
            f.write(str(i)+' ')
        f.write('\n')
    f.close()

if __name__ == '__main__':

    outputs = []
    li2 = [1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003]
    for p1 in range(0, 21):
            p1 /= 10.
        # for p2 in range(0, 5):
        #     p2 /= 10.

            processes = []
            for i in range(8):
                process = subprocess.Popen('CUDA_VISIBLE_DEVICES={} python -u train.py --epochs 200 --seed {}'.format(i, i), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                processes.append(process)
            output = [p.wait() for p in processes]
            output = [float(p.stdout.read().split(' ')[-3]) for p in processes]
            output = [p1] + output + [np.mean(output), np.std(output, ddof=1)]
            print(output)
            outputs.append(output)
            # output = [[0] for p in processes]
            # print(len(output), output)
            #output.append([p1, p2, tmp1, tmp2, tmp3, tmp4])

    writeFile('tmp', outputs)
