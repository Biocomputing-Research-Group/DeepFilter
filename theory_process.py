import pandas as pd
import shutil
import sys


def generate_sipros_input(file, outfile1, outfile2, outfilefasta):
    f = open(file)
    size = len(f.readlines())
    f.close()
    f = open(file)
    peptide = []
    unique_id = []
    fidx = open(outfile1, 'w')
    fcharge = open(outfile2, 'w')
    fpep = open(outfilefasta, 'w')
    for line_id, line in enumerate(f):
        if line_id == 0:
            continue
        line = line.strip().split(',')
        string = line[0]
        string = string.split('_')
        '''
        file_id=str(int(string[1]))
        scannum=str(int(string[2]))
        charge_id=str(int(string[3]))
        rank=str(int(string[4]))
        '''
        file_id = str(int(string[6]))
        scannum = str(int(string[7]))
        charge_id = str(int(string[8]))
        rank = str(int(string[9]))

        fidx.write(str(file_id) + '_' + str(scannum) +
                   '_' + str(charge_id) + '_' + str(rank))
        if line_id < size - 1:
            fidx.write('\n')
        peptide = line[26]
        s = peptide.replace('[15.9949]', '~')
        r = s.split('.')[1]
        fpep.write(r)
        if line_id < size - 1:
            fpep.write('\n')
        charge = charge_id
        fcharge.write(str(charge))
        if line_id < size - 1:
            fcharge.write('\n')

    fidx.close()
    fcharge.close()
    fpep.close()


# need to change by filename
def FeatureLoader(file, outfile):
    f = open(outfile, 'w')
    fr = open(file)
    for line_id, line in enumerate(fr):
        if line_id == 0:
            continue
        line = line.strip().split(',')
        string = line[0]
        string = string.split('_')
        '''
        file_id=str(int(string[1]))
        scan_id=str(int(string[2]))
        charge_id=str(int(string[3]))
        rank=str(int(string[4]))
        '''
        file_id = str(int(string[6]))
        scan_id = str(int(string[7]))
        charge_id = str(int(string[8]))
        rank = str(int(string[9]))

        idx = file_id + '_' + scan_id + '_' + charge_id + '_' + rank
        pep_str = line[26].replace('[15.9949]', '~').split('.')[1]
        # 11 feature from comet output
        Xcorr = line[9]
        deltLCn = line[6]
        deltCn = line[7]
        Mass = line[12]
        PepLen = line[13]
        charge1 = line[14]
        charge2 = line[15]
        charge3 = line[16]
        enzInt = line[22]
        dM = line[24]
        absdM = str(abs(float(line[3]) - float(line[4])))

        f.write(idx + '\n' + pep_str + '\n')
        f.write(Xcorr + ' ' + deltLCn + ' ' + deltCn + ' ' + Mass + ' ' + PepLen + ' ' +
                charge1 + ' ' + charge2 + ' ' + charge3 + ' ' + enzInt + ' ' + dM + ' ' + absdM + '\n\n')

    f.close()


'''
def ExpGenerate(file,msfile):
    f=open(msfile)
    fw=open('expMz.txt','w')
    fname=f.name.replace('.ms2','')
    file_id=str(fname.split('_')[3].replace('Repl',''))
    reader=pd.read_csv(file,sep='\t')
    for line_id,line in enumerate(reader['Scan number']):
    file_id=str(reader['Raw file'][line_id].split('_')[3].replace('Repl',''))
        scan=str(line.strip())
        idx=file_id+'_'+scan
'''


def main(input_file, out_file1, out_file2, out_file3, out_file4):
    # 1. extract comet output into sipros, require: unique_id(fileid_scanid), charge state, peptide string
    generate_sipros_input(input_file, out_file1, out_file2, out_file3)
    # 2. run sipros programm (c++)
    # 3. additional features loader
    FeatureLoader(input_file, out_file4)


if __name__ == "__main__":
    input_file = sys.argv[1]  # comet pin format
    out_file1 = sys.argv[2]  # idx
    out_file2 = sys.argv[3]  # charge
    out_file3 = sys.argv[4]  # peptide
    out_file4 = sys.argv[5]  # 11 feature
    main(input_file, out_file1, out_file2, out_file3, out_file4)
    # runing sipros
    # sipros command -i1 out_file1 -i2 out_file2 -i3 out_file3 -i4 theoratic.txt
