import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Merging fbank and pitch into single matrix
    # A simple implemetation of paste-feats
    fbank = np.load("mel_fbank_python.npy")
    pitch = np.load("pitch_python.npy")

    min_len = min(len(fbank), len(pitch))
    feature = np.append(fbank[:min_len, :], pitch[:min_len, :], axis=1)

    stats = np.load("cmvn.npy")
    dim = stats.shape[-1] - 1
    norm = np.zeros((2, dim))
    count = stats[0][dim]

    mean = stats[0, :dim] / count
    var = (stats[1, :dim] / count) - mean * mean
    var = np.clip(var, a_min=1e-20, a_max=None)
    scale = 1 / np.sqrt(var)
    if np.isnan(scale).any() or np.isinf(scale).any():
        raise Exception("NaN or infinity in cepstral mean/variance computation")
    offset = -(mean*scale)
    norm[0,:] = offset
    norm[1,:] = scale

    feature = feature * norm[1,:]
    feature += norm[0,:]

    print(feature.shape)
    np.save("normalized_fbank_and_pitch_python.npy", feature)
    for i in range(feature.shape[0]):
        for j in range(feature.shape[1]):
            print(f'{feature[i][j]:.7f} ', end='')
        print()

    # Plotting both kaldi and python values frame-by-frame for debug
    feat_kaldi = np.load("normalized_fbank_and_pitch.npy")
    vmax, vmin = np.max(feat_kaldi), np.min(feat_kaldi)
    fig = plt.figure(figsize=(40,20))
    plt.title("Normalized Feature from python")
    plt.imshow(np.rot90(feature), interpolation='nearest', aspect=1.4, vmax=vmax, vmin=vmin, cmap='binary')
    fig.savefig('Normalized Feature from python.png', bbox_inches='tight', format='png')
    plt.title("Normalized Feature from kaldi")
    plt.imshow(np.rot90(feat_kaldi), interpolation='nearest', aspect=1.4, vmax=vmax, vmin=vmin, cmap='binary')
    fig.savefig('Normalized Feature from kaldi.png', bbox_inches='tight', format='png')
    plt.close(fig)

    for i in range(len(feature)):
        fig = plt.figure()
        plt.plot(feature[i], label='python')
        plt.plot(feat_kaldi[i], label='kaldi')
        plt.legend()
        plt.show()
        fig.savefig(f"./norm_feat_plot/norm_feat_plot_#{i}.png", format='png')
        plt.close(fig)
