import sigpy.imaging as img
import numpy as np 

def deepMRInet(shape, acc=4, sample_n=10, centred=False):
    """
    Sampling mask from the original DeepMRI network. Adapted from the original
    implementation at https://github.com/js3611/Deep-MRI-Reconstruction/
    """
    def normal_pdf(length, sensitivity):
        return np.exp(-sensitivity * (np.arange(length) - length / 2) ** 2)

    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    pdf_x = normal_pdf(Nx, 0.5/(Nx/10.)**2)
    lmda = Nx/(2.*acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1./Nx

    if sample_n:
        pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1

    size = mask.itemsize
    mask = np.lib.stride_tricks.as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(shape)

    if not centred:
        mask = np.fft.ifftshift(mask, axes=(-1, -2))

    return mask.astype(np.bool)



# DeepMRINet
img.display(np.fft.fftshift(deepMRInet((368, 368), acc=3)), show=False, filename="deepMRInet_pattern_33.png")
img.display(np.fft.fftshift(deepMRInet((368, 368), acc=4)), show=False, filename="deepMRInet_pattern_25.png")
img.display(np.fft.fftshift(deepMRInet((368, 368), acc=6)), show=False, filename="deepMRInet_pattern_17.png")
