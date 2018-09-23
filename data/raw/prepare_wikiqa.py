import os


basedir = os.path.abspath('./WikiQACorpus/')
dstdir = os.path.abspath(os.path.dirname(__file__))
# infiles = ['WikiQA-train.txt',
#            'WikiQA-dev-filtered.txt',
#            'WikiQA-test-filtered.txt']
infiles = ['WikiQA-train.txt',
           'WikiQA-dev.txt',
           'WikiQA-test.txt']
infiles = [os.path.join(basedir, path) for path in infiles]
outfile = os.path.join(dstdir, 'raw_data.txt')

with open(outfile, 'w') as f:
    for infile in infiles:
        with open(infile, 'r') as f2:
            for line in f2:
                r = line.strip().split('\t')
                f.write('%s\t%s\t%s\n' % (r[0], r[1], r[2]))
