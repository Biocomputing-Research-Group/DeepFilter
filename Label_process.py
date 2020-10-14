import pandas as pd
import sys


class PSM:
    def __init__(self, PSM_Label, weight, file, scan, charge, rank, score, qvalue, pep, Peptide, Proteins):
        self.PSM_Label = PSM_Label
        self.weight = weight
        self.file = file
        self.scan = scan
        self.charge = charge
        self.rank = rank
        self.score = score
        self.qvalue = qvalue
        self.pep = pep
        self.Peptide = Peptide
        self.Proteins = Proteins


def merge_pin():
    record = []
    with open('OSU_D10_FASP_Elite_03202014_01.pin') as f:
        header = f.readline()
    for i in range(1, 10):
        with open('OSU_D10_FASP_Elite_03202014_0' + str(i) + '.pin') as f:
            for line_id, line in enumerate(f):
                if line_id == 0:
                    continue
                else:
                    record.append(line)

    for i in range(10, 12):
        with open('OSU_D10_FASP_Elite_03202014_' + str(i) + '.pin') as f:
            for line_id, line in enumerate(f):
                if line_id == 0:
                    continue
                else:
                    record.append(line)

    with open('marine1_all.pin', 'w') as f:
        f.write(header)
        for line in record:
            f.write(line)


def merge_TD(target, decoy):
    record = []
    with open(target) as f:
        for line in f:
            record.append(line)

    with open(decoy) as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                continue
            else:
                record.append(line)

    with open('marine1_all.csv', 'w') as f:
        for line in record:
            f.write(line)


def Label(file, outfile):
    PSMs = []
    with open(file) as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                continue
            else:
                s = line.strip().split('\t')
                length = len(s)
                score = float(s[1])
                qvalue = float(s[2])
                pep = float(s[3])
                Peptide = s[4]

                Proteins = []
                for pidx in range(5, length):
                    if s[pidx] is not '':
                        Proteins.append(s[pidx])
                PSM_Label = False
                for protein in Proteins:
                    if 'Rev' not in protein:
                        PSM_Label = True
                        break

            fileinfo = s[0].strip().split('_')
            file = int(fileinfo[5])
            scan = int(fileinfo[6])
            charge = int(fileinfo[7])
            rank = int(fileinfo[8])

            PSMs.append(PSM(PSM_Label, 0, file, scan, charge, rank,
                            score, qvalue, pep, Peptide, Proteins))

    rank_PSMs = sorted(PSMs, key=lambda psm: (psm.file, psm.scan, -psm.score))

    idx = str(rank_PSMs[0].file) + '_' + str(rank_PSMs[0].scan)
    last_scan = idx
    for psm_id, psm in enumerate(rank_PSMs):
        idx = str(psm.file) + '_' + str(psm.scan)
        freq_idx = 0
        if idx == last_scan:
            freq_idx += 1
        else:
            freq_idx = 0
        if freq_idx == 0:
            if psm.PSM_Label == True:
                psm.weight = 1 - psm.pep
            else:
                if psm.pep >= 0.5:
                    psm.weight = 0
                else:
                    psm.weight = (1 - psm.pep) / 10

        else:
            psm.PSM_Label = False
            psm.weight = 0
        last_scan = idx

    with open(outfile, 'w') as f:
        f.write('PSM_Label,weight,file,scan,charge,num,score,q_value,pep,peptide\n')
        for psm in rank_PSMs:
            f.write(str(psm.PSM_Label) + ',' + str(psm.weight) + ',' + str(psm.file) + ',' + str(psm.scan) + ',' + str(psm.charge) +
                    ',' + str(psm.rank) + ',' + str(psm.score) + ',' + str(psm.qvalue) + ',' + str(psm.pep) + ',' + str(psm.Peptide))
            f.write('\n')


if __name__ == "__main__":

    # mergedir=sys.argv[1]
    # merge_pin(mergedir)

    # run percolator

    target = sys.argv[1]  # target input
    decoy = sys.argv[2]  # decoy input
    outputPrefix = sys.argv[3]  # output file prefix
    filenum = sys.argv[4]  # number of output files
    merge_TD(target, decoy)
    Label('marine1_all.csv', 'Label_all_Relabel.csv')
    data = pd.read_csv('Label_all_Relabel.csv')
    for i in range(1, int(filenum) + 1):
        subdata = data[data['file'] == i]
        subdata.to_csv(outputPrefix + str(i) + '.csv', index=False)
