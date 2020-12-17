from libs.inference import get_parser
from libs.inference import SpeechRecognizer
from scipy.io.wavfile import read
import sys
import numpy


def predict(filename=None,AR=True):
    parser=get_parser()
    args=parser.parse()
    args.model='./model.pt'
    if AR is False: #Trandformer 인코더만 사용하여 결과 출력, 인식률 하락, 속도 20배
        args.fast_decode=True
    a = read(args.filename)
    data = numpy.array(a[1],dtype=numpy.int16)
    asr=SpeechRecognizer(args)
    ret=asr.recog_buffer(data)
    return ret

if __name__ == "__main__":
    print(sys.argv)
    ret=predict(sys.argv[1])
    print(ret)
