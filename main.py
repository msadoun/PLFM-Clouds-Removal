from models.PLFM import PLFM
import sys

# Suppress warnings
import logging
import os
import warnings
import rasterio
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def main(argv):
    if len(argv) == 0:
        print('[!] Usage: python main.py --train <dataset_path> | --test <dataset_path> [output_dir]')
        return

    # Initialize PLFM
    print('[-] Initializing PLFM \n')
    plfm = PLFM('weights')

    # Training PLFM
    if argv[0] == '--train':
        print('[-] PLFM Training\n')

        if len(argv)>1:
            plfm.train(argv[1])
        else:
            plfm.train('SeriesSen1-2')
    # Testing PLFM
    elif argv[0] == '--test':
        if len(argv)>1:
            if len(argv) > 2:
                plfm.test(argv[1], argv[2])
            else:
                plfm.test(argv[1])
        else:
            plfm.test('SeriesSen1-2')
    else:
        print('[!] Error, function {} not found!!!'.format(argv[0]))
    

if __name__ == '__main__':
    main(sys.argv[1:])
