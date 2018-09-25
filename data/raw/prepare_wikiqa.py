import os

WIKIQA_URL = "https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip"


def prepare_wikiQA():
    basedir = os.path.abspath('./WikiQACorpus/')
    dstdir = os.path.abspath(os.path.dirname(__file__))
    # infiles = ['WikiQA-train.txt',
    #            'WikiQA-dev-filtered.txt',
    #            'WikiQA-test-filtered.txt']
    infiles = ['WikiQA-train.txt',
               'WikiQA-dev.txt',
               'WikiQA-test.txt']
    infiles = [os.path.join(basedir, path) for path in infiles]
    outfile = os.path.join(dstdir, 'raw_wiki_data.txt')

    try:
        with open(outfile, 'w') as f:
            for infile in infiles:
                with open(infile, 'r') as f2:
                    for line in f2:
                        r = line.strip().split('\t')
                        f.write('%s\t%s\t%s\n' % (r[0], r[1], r[2]))
    except FileExistsError as e:
        error_msg = f"FileExistsError, {e} not found." + \
            "Does the WikiQACorpus file exists?\n" + \
            "If not run:\ncd data/raw/\n" + \
            f"wget {WIKIQA_URL}\nunzip WikiQACorpus.zip"
        print(error_msg)


if __name__ == '__main__':
    prepare_wikiQA()
