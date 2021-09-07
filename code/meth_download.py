import pandas as pd
import os
import sys
import subprocess

# totale = 1141

meth_file = './meth_manifest_preproc.txt'
meth = pd.read_csv(meth_file, delimiter = "\t", engine='python')
list_ids = meth['id']
list_filename = meth['filename']
check_index = 0

for uuid, filename in zip(list_ids,list_filename):

    try:
        if not os.path.exists(os.path.join(os.getcwd(), uuid)):

            # gdc-client download uui
            cmd = "./gdc-client download {}".format(uuid)
            os.system(cmd)

            # read content /uuid/filename.txt
            path = "./{}/{}".format(uuid, filename)
            df = pd.read_csv(path, delimiter = "\t")
            # keep 2 cols
            # Composite Element REF (1 col)
            # Beta_value (2 col)
            df = df[['Composite Element REF', 'Beta_value']]
            assert(df.shape[1] == 2), "check shape"
            # print(df)
            # delete previous file
            os.system("rm --force {}".format(path))
            os.system("rm -r --force ./{}/logs/".format(uuid))
            # save file
            df.to_csv(path, header=True, index=False, sep='\t')
            print(uuid, "done")
            check_index += 1
        else:
            print(uuid, "already exists")
    except:
        f = open("checkpoint.txt", "a")
        output = "{} {} {}\n".format(check_index, uuid, filename)
        f.write(output)
        f.close()
        #sys.exit()